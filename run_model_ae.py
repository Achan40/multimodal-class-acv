import os
import argparse
import numpy as np
import torch

from torch.utils.data import  DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.modeling_ae import Autoencoder

from helper import DataImageOnly, load_pkl, compute_auroc, item_preds_img_only, tracking_results_file, write_to_tracking_results, disease_list

# Hyperparameters
input_dim = 244 # shape of flattened image
encoding_dim = 32

def train():
    img_dir = args.DATA_DIR

    # Creating an IRENE model, or using a saved checkpoint if one is indicated.
    if args.SAVED_MOD is None:
        print("No saved model specified. Training new model.")
        # create model object and optimizer
        model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # define loss function
    loss_fn = torch.nn.MSELoss()

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    # --------------- Create or use existing csv for tracking training metrics-------------------
    # headers default to: ["dataset", "set_num","epoch", "trn_loss", "val_loss"]
    tracking_results_file(filename='./checkpoints/metrics.csv', headers=["dataset", "set_num","epoch", "trn_loss", "val_loss"])

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
        data = DataImageOnly(pkl_train_dict, img_dir, transform=data_transforms['test'])
        loader = DataLoader(data, batch_size=args.BSZ, shuffle=False, num_workers=12, pin_memory=True)


        # Open validation .pkl file and save the dict to a variable
        pkl_val_dict = load_pkl(args.VAL_LAB_SET)

        # Create Validation Dataset and Dataloader object
        val_data = DataImageOnly(pkl_val_dict, img_dir, transform=data_transforms['test'])
        val_loader = DataLoader(val_data, batch_size=args.BSZ, shuffle=False, num_workers=12, pin_memory=True)

        print('Training on set '+str(set_num)+': ', pkl_file)

        for epoch in range(args.EPCHS):
            #---------------------- Begin Training----------------------------
            model.train()

            # track train loss per epoch
            trn_loss = 0.0

            # track metrics
            metrics_arr = []

            for item in tqdm(loader):
                inputs, _ = item
                # send input to gpu if available
                inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                # outputs
                outputs = model(inputs)

                # calculate loss for each batch
                loss = loss_fn(outputs, inputs)

                # add to total loss
                trn_loss += loss.item()

                optimizer.zero_grad()
                # with optimizer.scale_loss(loss) as scaled_loss:
                #     scaled_loss.backward()

                loss.backward()
                optimizer.step()
            
            # calculate average loss across all batches
            trn_loss /= len(loader)
            
            print(f"Epoch {epoch+1}/{args.EPCHS}, "
            f"Training Loss: {trn_loss:.4f}, ")

            # add values to array for metrics tracking. 
            # name of dataset, epoch number, training loss
            metrics_arr.extend([pkl_file, set_num, epoch+1, trn_loss])

            #---------------------- Begin Validation----------------------------
            model.eval()

            # track validation loss
            val_loss = 0.0

            with torch.no_grad():
                
                for item in tqdm(val_loader):
                    inputs, _ = item
                    # send input to gpu if available
                    inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                    # outputs
                    outputs = model(inputs)

                    # calculate loss
                    val_loss += loss_fn(outputs, inputs).item()

            # calculate average validation loss across all batches
            val_loss /= len(val_loader)


            print(f"Epoch {epoch+1}/{args.EPCHS}, "
            f"Validation Loss: {val_loss:.4f}, ")

            # add values to array for metrics tracking. 
            # validation loss and validation mean auroc
            metrics_arr.extend([val_loss])

            # write a new line to the tracking file every epoch
            write_to_tracking_results(metrics_arr)

            torch.cuda.empty_cache()

            # -------------- Saving the Model after training on each epoch-----------
            path = './checkpoints'
            os.makedirs(path, exist_ok=True)
            path = './checkpoints/set_'+str(set_num)+'_epc_'+str(epoch+1)+'.pt'
            torch.save(model, path)

        set_num += 1

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

    img_dir = args.DATA_DIR

    pkl_test_dict = load_pkl(args.TST_LAB_SET)

    # Create Train Dataset and DataLoader object
    test_data = DataImageOnly(pkl_test_dict, img_dir, transform=data_transforms['test'])
    test_loader = DataLoader(test_data, batch_size=args.BSZ, shuffle=False, num_workers=12, pin_memory=True)

    # load a saved model
    model = torch.load(args.SAVED_MOD)

    if torch.cuda.is_available():
        model.cuda()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    # Using nvidia apex for optimization
    #model, optimizer = amp.initialize(model.cuda(), optimizer, opt_level="O1")

    # define loss function
    loss_fn = torch.nn.MSELoss()

    #---------------------- Begin Testing----------------------------
    model.eval()

    # track test loss
    test_loss = 0.0 

    with torch.no_grad():
        
        for item in tqdm(test_loader):
            inputs, _ = item
            inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # outputs
            outputs = model(inputs)

            # calculate loss
            test_loss += loss_fn(outputs, inputs).item()

    print(f"Test Loss: {test_loss:.4f} ")
    # calculate average validation loss across all batches
    test_loss /= len(test_loader)
         

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