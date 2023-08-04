import os
import pickle
import argparse
from multiprocessing import Manager
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from models.modeling_irene import IRENE, CONFIGS

# token limit for unstructured textual data
TK_LIM = 40

disease_list = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices', 'No Finding']

class Data(Dataset):
    '''
    Takes in a sturctured data formatted as a python dictionary,
    path to a directory of images, and any tranformations that need to be performed
    '''
    def __init__(self, d_set, img_dir, transform=None, target_transform=None):
        ''' Wrap dicts in Manager object. This deals with the copy-on-access problem of 
        forked python processes since we use a standard python dict for our data object. 
        If we don't do this, the num_workers parameter 
        in our dataloader object will duplicate memory for each worker. 
        Note: There is an issue in windows with multiprocessing where the 
        Manager() cannot handle large dictionaries. 
        Had to run in linux to run train without issue
        '''
        manager = Manager()
        self.mm_data = manager.dict(d_set)
        self.idx_list = manager.list(self.mm_data.keys())
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        k = self.idx_list[idx]
        img_path = os.path.join(self.img_dir, k)
        img = Image.open(img_path).convert('RGB')

        label = self.mm_data[k]['label'].astype('float32')
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        cc = torch.from_numpy(self.mm_data[k]['pdesc']).float()
        demo = torch.from_numpy(np.array(self.mm_data[k]['bics'])).float()
        lab = torch.from_numpy(self.mm_data[k]['bts']).float()
        return img, label, cc, demo, lab

def load_pkl(dict_path):
    '''
    Takes in a path to a .pkl file
    and saves the data as a variable
    '''
    f = open(dict_path, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def compute_auroc(data_gt, data_pred, classCount=14):
    '''
    Comput auroc values using true values and
    probabilites for each class predicition
    '''
    out_auroc = []
    data_np_gt = data_gt.cpu().numpy()
    data_np_pred = data_pred.cpu().numpy()

    for i in range(classCount):
        # If is only one unique value in a column of our true labels, append NA.
        # Cannot compute roc_auc otherwise
        if len(np.unique(data_np_gt[:,i])) <= 1:
            out_auroc.append(np.nan)
        else:
            out_auroc.append(roc_auc_score(data_np_gt[:, i], data_np_pred[:, i]))
        return out_auroc

def item_preds(item, model):
    '''
    Takes in a list(batch) of items and the model.
    Then generate predictions and return the predictions and true labels for each
    '''
    imgs, labels, cc, demo, lab = item

    imgs = imgs.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    cc = cc.view(-1, TK_LIM, cc.shape[3]).cuda(non_blocking=True).float()
    demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
    lab = lab.view(-1, lab.shape[1], 1).cuda(non_blocking=True).float()
    sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
    age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()

    preds = model(imgs, cc, lab, sex, age)[0]

    return preds, labels

def train():
    '''
    Function for training IRENE.
    Parameters for this function comes from the passed in args
    '''

    num_classes = len(disease_list)
    config = CONFIGS["IRENE"]
    img_dir = args.DATA_DIR

    # Creating an IRENE model, or using a saved checkpoint if one is indicated.
    if args.SAVED_MOD is None:
        print("No saved model indicated. Training new model.")
        # create model object and optimizer
        model = IRENE(config, 224, zero_head=True, num_classes=num_classes)
        set_num = 0
    else:
        print("Continue training model: " + args.SAVED_MOD)
        model = torch.load(args.SAVED_MOD)

        # Hit conditional if SAVED_MOD is specified and SAVED_MOD_OFFSET is specified
        if args.SAVED_MOD_OFFSET is None:
            print("No offset indicated. May overwrite checkpoints when saving. Beginning at offset 0")
            set_num = 0
        # Tracking the set number
        else:
            print("Offset specified. Will begin saving model at offset: " + str(args.SAVED_MOD_OFFSET))
            set_num = args.SAVED_MOD_OFFSET

    if torch.cuda.is_available():
        model.cuda()
    
    # build the optimizer
    optimizer_irene = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    # define loss function
    loss_fn = torch.nn.BCELoss()

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }


    '''
    args.TRN_LAB_SET can be a list of pkl files.
    This way, we can train over the entire dataset after splitting it into parts.
    Would require a very large amount of memory otherwise. Also note 
    that in windows, the multiprocessor package will throw errors if you load too 
    much data into memory at once. Limit the size of your .pkl files, otherwise you will
    have to train on a Linux based machine. 
    '''
    for pkl_file in args.TRN_LAB_SET:
        pkl_train_dict = load_pkl(pkl_file)

        # Create Train Dataset and DataLoader object 
        data = Data(pkl_train_dict, img_dir, transform=data_transforms['test'])
        loader = DataLoader(data, batch_size=args.BSZ, shuffle=False, num_workers=12, pin_memory=True)


        # Open validation .pkl file and save the dict to a variable
        pkl_val_dict = load_pkl(args.VAL_LAB_SET)

        # Create Validation Dataset and Dataloader object
        val_data = Data(pkl_val_dict, img_dir, transform=data_transforms['test'])
        val_loader = DataLoader(val_data, batch_size=args.BSZ, shuffle=False, num_workers=12, pin_memory=True)

        

        # Using nvidia apex for optimization
        #model, optimizer_irene = amp.initialize(model.cuda(), optimizer_irene, opt_level="O1")

        print('Training on set '+str(set_num)+': ', pkl_file)

        for epoch in range(args.EPCHS):
            #---------------------- Begin Training----------------------------
            model.train()

            # Initialize variables used to calculate AUROC
            outGT = torch.FloatTensor().cuda(non_blocking=True)
            outPRED = torch.FloatTensor().cuda(non_blocking=True)

            # track train loss per epoch
            trn_loss = 0.0
            
            for item in tqdm(loader):
                # get the inputs; data is a list of [inputs, labels]
                preds, labels = item_preds(item=item, model=model)

                # probability values
                probs = torch.sigmoid(preds)

                # calculate loss for each batch
                loss = loss_fn(probs, labels)

                # add to total loss
                trn_loss += loss.item()

                outGT = torch.cat((outGT, labels), 0)
                outPRED = torch.cat((outPRED, probs.data), 0)

                optimizer_irene.zero_grad()
                # with optimizer_irene.scale_loss(loss) as scaled_loss:
                #     scaled_loss.backward()

                loss.backward()
                optimizer_irene.step()
            
            # calculate average loss across all batches
            trn_loss /= len(loader)

            # calculate AUROC
            aurocIndividual = compute_auroc(outGT, outPRED, classCount=num_classes)
            aurocMean = np.nanmean(np.array(aurocIndividual))
            
            print(f"Epoch {epoch+1}/{args.EPCHS}, "
            f"Training Loss: {trn_loss:.4f}, ")

            print('mean AUROC:' + str(aurocMean))

            #---------------------- Begin Validation----------------------------
            model.eval()

            # track validation loss
            val_loss = 0.0

            with torch.no_grad():

                # Initialize variables used to calculate AUROC
                outGT = torch.FloatTensor().cuda(non_blocking=True)
                outPRED = torch.FloatTensor().cuda(non_blocking=True)
                
                for item in tqdm(val_loader):
                    preds, labels = item_preds(item=item, model=model)

                    # probability values
                    probs = torch.sigmoid(preds)

                    # calculate loss
                    val_loss += loss_fn(probs, labels).item()

                    outGT = torch.cat((outGT, labels), 0)
                    outPRED = torch.cat((outPRED, probs.data), 0)

            # calculate average validation loss across all batches
            val_loss /= len(val_loader)

            # calculate AUROC
            aurocIndividual = compute_auroc(outGT, outPRED, classCount=num_classes)
            aurocMean = np.nanmean(np.array(aurocIndividual))
            
            # show auroc for each class
            #for i in range (0, len(aurocIndividual)):
                #print(disease_list[i] + ': '+str(aurocIndividual[i]))

            print(f"Epoch {epoch+1}/{args.EPCHS}, "
            f"Validation Loss: {val_loss:.4f}, ")

            print('Mean AUROC:' + str(aurocMean))
            torch.cuda.empty_cache()
        
        # -------------- Saving the Model after training on each set-----------
        path = './checkpoints'
        os.makedirs(path, exist_ok=True)
        path = './checkpoints/set_'+str(set_num)+'mod.pt'
        torch.save(model, path)

        # Clean objects from mem when finished training on a set
        del data, loader, val_data, val_loader

def test():
    '''
    Function for testing our saved model.
    Parameters for this function comes from the passed in args
    '''
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    num_classes = len(disease_list)
    img_dir = args.DATA_DIR

    # Create Train Dataset and DataLoader object
    test_data = Data(args.TST_LAB_SET, img_dir, transform=data_transforms['test'])
    test_loader = DataLoader(test_data, batch_size=args.BSZ, shuffle=False, num_workers=12, pin_memory=True)

    # load a saved model
    model = torch.load(args.SAVED_MOD)

    if torch.cuda.is_available():
        model.cuda()
    #optimizer_irene = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    # Using nvidia apex for optimization
    #model, optimizer_irene = amp.initialize(model.cuda(), optimizer_irene, opt_level="O1")

    #---------------------- Begin Testing----------------------------
    model.eval()
    with torch.no_grad():

        # Initialize variables used to calculate AUROC
        outGT = torch.FloatTensor().cuda(non_blocking=True)
        outPRED = torch.FloatTensor().cuda(non_blocking=True)
        
        for item in tqdm(test_loader):
            preds, labels = item_preds(item=item, model=model)

            # probability values
            probs = torch.sigmoid(preds)

            outGT = torch.cat((outGT, labels), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)

    # calculate AUROC
    aurocIndividual = compute_auroc(outGT, outPRED, classCount=num_classes)
    aurocMean = np.nanmean(np.array(aurocIndividual))
    
    # show auroc for each class
    for i in range (0, len(aurocIndividual)):
        print(disease_list[i] + ': '+str(aurocIndividual[i]))

    print('Mean AUROC:' + str(aurocMean))
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for model training and testing")
    subparsers = parser.add_subparsers(dest="subcommand", help="Choose -train or -test")

    # positional arguements required for both train and test functions. 
    parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int) # batch size
    parser.add_argument('--DATA_DIR', action='store', dest='DATA_DIR', required=True, type=str) # parent directory to your train, valid, test image folders

    # conditional arguements based on whether we want to train the model or make predictions on a test set
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--TRN_LAB_SET', nargs='+', action='store', dest='TRN_LAB_SET',help='List of pkl files to use for training', required=True, type=str) # path to train.pkl file. Can pass in multiple. This will allow us to train the large dataset without consuming too much system memory
    parser_train.add_argument('--VAL_LAB_SET', action='store', dest='VAL_LAB_SET', required=True, type=str) # path to valid.pkl file
    parser_train.add_argument('--EPCHS', action='store', dest='EPCHS', required=True, type=int) # number of epochs
    parser_train.add_argument('--SAVED_MOD', action='store', dest='SAVED_MOD', required=False, type=str) # use if you have an existing model and want to continue training
    parser_train.add_argument('--SAVED_MOD_OFFSET', action='store', dest='SAVED_MOD_OFFSET', required=False, type=int) # Specify the offset so you don't overwrite a checkpoint

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--TST_LAB_SET', action='store', dest='TST_LAB_SET', required=True, type=str) # path to test.pkl file
    parser_test.add_argument('--SAVED_MOD', action='store', dest='SAVED_MOD', required=True, type=str) # path to test.pkl file

    args = parser.parse_args()

    if args.subcommand == 'train':
        #./data/actual/train.pkl ./data/actual/train2.pkl ./data/actual/train3.pkl ./data/actual/train4.pkl ./data/actual/train5.pkl ./data/actual/train6.pkl ./data/actual/train7.pkl ./data/actual/train8.pkl'
        train()
    elif args.subcommand == 'test':
        test()
    else:
        print('No -train or -test flag provided')