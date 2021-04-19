from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
import os
IMG_FOLDER_NAME = "JPEGImages"

def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy',allow_pickle=True).item()


    return [cls_labels_dict[img_name] for img_name in img_name_list]

def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    return img_name_list

def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

class VOC_Dataset(Dataset):
    def __init__(self,img_name_list_path,voc12_root,transform=None):
        self.label_list = load_image_label_list_from_npy(load_img_name_list(img_name_list_path))
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc_root = voc12_root
        self.transform = transform

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = Image.open(get_img_path(name,self.voc_root)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.from_numpy(self.label_list[idx])

        return name,img,label

    def __len__(self):
        return len(self.img_name_list)
