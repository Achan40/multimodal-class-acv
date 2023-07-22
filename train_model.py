import os
import pickle
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from apex import amp

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from models.modeling_irene import IRENE, CONFIGS
from torchvision import transforms, utils
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# token limit for unstructured textual data
tk_lim = 40

disease_list = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices', 'No Finding']

class Data(Dataset):
    def __init__(self, set_type, img_dir, transform=None, target_transform=None):
        dict_path = set_type
        f = open(dict_path, 'rb') 
        self.mm_data = pickle.load(f)
        f.close()
        self.idx_list = list(self.mm_data.keys())
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
    
# Use to check if our .pkl files were created correctly
def check_pkl(dict_path):
    f = open(dict_path, 'rb') 
    data = pickle.load(f)
    f.close()
    return data

#  computing AUROC values
def computeAUROC (dataGT, dataPRED, classCount=14):
    outAUROC = []
        
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
        
    for i in range(classCount):
        # If is only one unique value in a column of our true labels, append NA, otherwise roc_auc cannot be computed
        if len(np.unique(datanpGT[:,i])) <= 1:
            outAUROC.append(np.nan)
        else:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
    return outAUROC

# return predictions and labels for some item in our dataloader object
def item_preds(item, model):
    imgs, labels, cc, demo, lab = item

    imgs = imgs.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    cc = cc.view(-1, tk_lim, cc.shape[3]).cuda(non_blocking=True).float()
    demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
    lab = lab.view(-1, lab.shape[1], 1).cuda(non_blocking=True).float()
    sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
    age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()

    preds = model(imgs, cc, lab, sex, age)[0]

    return preds, labels

# function for loading weights
def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    print("Loading Model...")
    return model

def train():
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    num_classes = len(disease_list)
    config = CONFIGS["IRENE"]
    img_dir = args.DATA_DIR

    # Create Train Dataset and DataLoader object
    data = Data(args.TRN_LAB_SET, img_dir, transform=data_transforms['test'])
    loader = DataLoader(data, batch_size=args.BSZ, shuffle=False, num_workers=16, pin_memory=True)

    # Create Validation Dataset and Dataloader object
    val_data = Data(args.VAL_LAB_SET, img_dir, transform=data_transforms['test'])
    val_loader = DataLoader(val_data, batch_size=args.BSZ, shuffle=False, num_workers=16, pin_memory=True)

    # create model object and optimizer
    model = IRENE(config, 224, zero_head=True, num_classes=num_classes)

    if torch.cuda.is_available():
        model.cuda()
    optimizer_irene = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    # Using nvidia apex for optimization
    #model, optimizer_irene = amp.initialize(model.cuda(), optimizer_irene, opt_level="O1")

    # define loss function
    loss_fn = torch.nn.BCELoss()

    for epoch in range(args.EPCHS):
        #---------------------- Begin Training----------------------------
        model.train()

        # Initialize variables used to calculate AUROC
        outGT = torch.FloatTensor().cuda(non_blocking=True)
        outPRED = torch.FloatTensor().cuda(non_blocking=True)
        
        for item in tqdm(loader):
            # get the inputs; data is a list of [inputs, labels]
            preds, labels = item_preds(item=item, model=model)

            # probability values
            probs = torch.sigmoid(preds)

            # calculate loss
            loss = loss_fn(probs, labels)

            outGT = torch.cat((outGT, labels), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)

            optimizer_irene.zero_grad()
            # with optimizer_irene.scale_loss(loss) as scaled_loss:
            #     scaled_loss.backward()

            loss.backward()
            optimizer_irene.step()

        # calculate AUROC
        aurocIndividual = computeAUROC(outGT, outPRED, classCount=num_classes)
        aurocMean = np.nanmean(np.array(aurocIndividual))
        
        print(f"Epoch {epoch+1}/{args.EPCHS}, "
        f"Training Loss: {loss.item():.4f}, ")

        print('mean AUROC:' + str(aurocMean))

        # -------------- Saving the Model at Every Epoch-----------
        path = './checkpoints'
        os.makedirs(path, exist_ok=True)
        path = './checkpoints/mod'+str(epoch)+'.pt'
        torch.save(model, path)

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
        aurocIndividual = computeAUROC(outGT, outPRED, classCount=num_classes)
        aurocMean = np.nanmean(np.array(aurocIndividual))
        
        # show auroc for each class
        #for i in range (0, len(aurocIndividual)):
            #print(disease_list[i] + ': '+str(aurocIndividual[i]))

        print(f"Epoch {epoch+1}/{args.EPCHS}, "
        f"Validation Loss: {val_loss:.4f}, ")

        print('Mean AUROC:' + str(aurocMean))

def test():
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
    test_loader = DataLoader(test_data, batch_size=args.BSZ, shuffle=False, num_workers=16, pin_memory=True)

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
    aurocIndividual = computeAUROC(outGT, outPRED, classCount=num_classes)
    aurocMean = np.nanmean(np.array(aurocIndividual))
    
    # show auroc for each class
    for i in range (0, len(aurocIndividual)):
        print(disease_list[i] + ': '+str(aurocIndividual[i]))

    print('Mean AUROC:' + str(aurocMean))
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int) # batch size
    parser.add_argument('--EPCHS', action='store', dest='EPCHS', required=True, type=int) # number of epochs

    parser.add_argument('--DATA_DIR', action='store', dest='DATA_DIR', required=True, type=str) # parent directory to your train, valid, test image folders
    parser.add_argument('--TRN_LAB_SET', action='store', dest='TRN_LAB_SET', required=False, type=str) # path to train.pkl file
    parser.add_argument('--VAL_LAB_SET', action='store', dest='VAL_LAB_SET', required=False, type=str) # path to valid.pkl file
    parser.add_argument('--TST_LAB_SET', action='store', dest='TST_LAB_SET', required=False, type=str) # path to test.pkl file

    parser.add_argument('--SAVED_MOD', action='store', dest='SAVED_MOD', required=False, type=str) # path to test.pkl file

    args = parser.parse_args()

    test()

    # data_transforms = {
    #         'test': transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #         ]),
    #     }
    # img_dir = args.DATA_DIR
    # val_data = Data(args.VAL_LAB_SET, img_dir, transform=data_transforms['test'])
    
    # print(val_data.idx_list)

    # Showing the image
    # data_iter = iter(loader)
    # images = next(data_iter)

    # images_np = np.array(images[0][0])

    # def show_image(image_np):
    #     plt.imshow(image_np)
    #     plt.axis('off')
    #     plt.show()

    # Assuming you want to show the first image in the batch
    # show_image(images_np[0])

    # print(data_tmp.idx_list)
    # print(data_tmp.mm_data['patient00001/study1/view1_frontal'])
    # print(data_tmp['patient00001/study1/view1_frontal'])