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
    
def train_model():
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    num_classes = args.CLS
    config = CONFIGS["IRENE"]
    img_dir = args.DATA_DIR

    # token limit for unstructured textual data
    tk_lim = 40

    # Create Dataset and DataLoader object
    data = Data(args.SET_TYPE, img_dir, transform=data_transforms['test'])
    loader = DataLoader(data, batch_size=args.BSZ, shuffle=False, num_workers=16, pin_memory=True)

    # create model object and optimizer
    model = IRENE(config, 224, zero_head=True, num_classes=num_classes)

    if torch.cuda.is_available():
        model.cuda()

    optimizer_irene = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    # Using nvidia apex for optimization
    #model, optimizer_irene = amp.initialize(model.cuda(), optimizer_irene, opt_level="O1")

    # define loss function
    loss_fn = torch.nn.BCELoss()

    model.train()
    for epoch in range(5):
        for item in tqdm(loader):
        # get the inputs; data is a list of [inputs, labels]
            imgs, labels, cc, demo, lab = item

            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            cc = cc.view(-1, tk_lim, cc.shape[3]).cuda(non_blocking=True).float()
            demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
            lab = lab.view(-1, lab.shape[1], 1).cuda(non_blocking=True).float()
            sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
            age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()

            preds = model(imgs, cc, lab, sex, age)[0]

            # probability values
            probs = torch.sigmoid(preds)

            loss = loss_fn(probs, labels)

            optimizer_irene.zero_grad()
            # with optimizer_irene.scale_loss(loss) as scaled_loss:
            #     scaled_loss.backward()

            loss.backward()
            optimizer_irene.step()

        print("Training Loss: ",loss.item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--CLS', action='store', dest='CLS', required=True, type=int)
    parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int)
    parser.add_argument('--DATA_DIR', action='store', dest='DATA_DIR', required=True, type=str)
    parser.add_argument('--SET_TYPE', action='store', dest='SET_TYPE', required=True, type=str)
    args = parser.parse_args()
    
    train_model()

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