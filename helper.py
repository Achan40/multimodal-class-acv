import pickle
import os
import numpy as np
import csv
import torch

from sklearn.metrics import roc_auc_score
from PIL import Image
from torch.utils.data import Dataset
from multiprocessing import Manager

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
    
class DataImageOnly(Data):
    '''
    Inherits from our general Data class.
    Overwrites the __getitem__ method to only use img and label data.
    '''
    def __init__(self, d_set, img_dir, transform=None, target_transform=None):
        super().__init__(d_set, img_dir, transform, target_transform)
    
    def __getitem__(self, idx):
        k = self.idx_list[idx]
        img_path = os.path.join(self.img_dir, k)
        img = Image.open(img_path).convert('RGB')

        label = self.mm_data[k]['label'].astype('float32')
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, label

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
    Compute auroc values using true values and
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

def item_preds_img_only(item, model):
    '''
    Takes in a list(batch) of items and the model.
    Then generate predictions and return the predictions and true labels for each.
    This function only uses the image data and the model as inputs.
    '''
    imgs, labels = item

    imgs = imgs.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)

    preds = model(imgs)

    return preds, labels


def tracking_results_file(filename='./checkpoints/metrics.csv', headers=["dataset", "set_num","epoch", "trn_loss", "trn_mean_auroc", "val_loss", "val_mean_auroc"]):
    '''
    Create a file for tracking various metrics while training
    '''
    # Check if the file exists
    if not os.path.exists(filename):
        # If the file doesn't exist, create it and write the headers
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    else:
        # If the file already exists, do nothing
        print("metrics.csv already exists. Will continue writing to file.")

def write_to_tracking_results(arr, filename='./checkpoints/metrics.csv'):
    '''
    Write to an existing tracking file. The tracking file should already be
    created before this function is called
    '''
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(arr)