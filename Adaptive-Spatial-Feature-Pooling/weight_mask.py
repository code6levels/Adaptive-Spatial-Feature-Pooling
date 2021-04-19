import torch
import torch.optim as optim
from network import resnet,HRnet,PB_resnet,PB_net
from tool.dataset import VOC_Dataset
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
from tool.utils import save_cam,save_mask
import numpy as np
from evaluation import do_python_eval
from tqdm import tqdm
import yaml
voc_root = '/users4/mxtuo/zhanghan/data/VOC2012'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--lr',default=1e-4,type=float)
    parser.add_argument('--gpu_index',default='2',type=str)
    parser.add_argument('--crop_size',default=224,type=int)
    parser.add_argument('--num_epochs',default=7,type=int)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)


    args = parser.parse_args()
    print(args)
    import os
    # os.makedirs(args.session_name,exist_ok=True)
    # os.makedirs(f'{args.session_name}/model_weights', exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index



    val_dataset = VOC_Dataset(args.val_list,voc_root,transform = transforms.Compose([
            transforms.Resize(args.crop_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) )
    # loglist = do_python_eval(f'baseline_3_5/pred_dir', args, 0)


    val_dataloader = DataLoader(val_dataset,batch_size=32,shuffle=False,num_workers=4,pin_memory=False,drop_last=True)

    test_dataset = VOC_Dataset(args.val_list,voc_root,transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) )
    test_dataloader = DataLoader(test_dataset,shuffle=False,num_workers=8,pin_memory=True)



    model = PB_resnet.net(20)
    model.load_state_dict(torch.load("/users4/mxtuo/zhanghan/DBnet/model_weights/res50-bs=96-cf=20.0-mIoU=47.811.pth"))
    weight = model.fc.weight[0].view(-1)
    print(torch.topk(weight,int(2048*0.1)))
