import torch
import torch.optim as optim
from network import resnet,PB_resnet,PB_net
from tool.dataset import VOC_Dataset
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
from tool.utils import save_cam,save_mask,save_cam_2
import numpy as np
from evaluation import do_python_eval
from tqdm import tqdm
voc_root = '/home/liuyang/zhanghan/VOCdevkit/VOC2012'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--lr',default=1e-4,type=float)
    parser.add_argument('--gpu_index',default='1',type=str)
    parser.add_argument('--crop_size',default=224,type=int)
    parser.add_argument('--num_epochs',default=7,type=int)
    parser.add_argument('--session_name',required=True,type=str)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument('--gpu_num',default=1,type=int)
    parser.add_argument('--COEFF',default=12.0,type=float)
    parser.add_argument('--weight_decay',default=0,type=float)
    parser.add_argument('--bg_weight',default=0.2,type=float)

    args = parser.parse_args()
    print(args)
    import os
    os.makedirs(args.session_name,exist_ok=True)
    os.makedirs(f'{args.session_name}/model_weights', exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index


    test_dataset = VOC_Dataset(args.val_list,voc_root,transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) )
    test_dataloader = DataLoader(test_dataset,shuffle=False,num_workers=8,pin_memory=True)



    # model = PB_resnet.net(args.COEFF).cuda()
    model = PB_resnet.net(args.COEFF).cuda()
    model.load_state_dict(torch.load("/home/liuyang/zhanghan/PBnet/Lip_50_cf=20_bs=96_lr=0.0001_wd=6e-05/model_weights/res50-bs=96-cf=20.0-mIoU=47.811.pth"))
    if(args.gpu_num>1):
        model = nn.DataParallel(model)


    model.eval()


    for name, img, label in tqdm(test_dataloader):
        img = img.cuda()
        with torch.set_grad_enabled(False):
            cam, _, = model(img)
        b, c, h, w = cam.shape
        cam = F.relu(cam).cpu()
        cam_max = torch.max(cam.view(b, c, -1), dim=-1)[0].view(b, c, 1, 1) + 1e-5
        cam = F.relu(cam - 1e-5, inplace=True) / cam_max
        cam = cam * (label.view(b, c, 1, 1))
        cam = F.interpolate(cam, img.shape[2:], mode='bilinear', align_corners=False).squeeze()
        bg_socre = (torch.ones(img.shape[2:]) * args.bg_weight).unsqueeze(0)
        pred = torch.argmax(torch.cat((bg_socre, cam), dim=0), dim=0).numpy().astype(np.uint8)
        save_mask(pred, args, name[0])
    loglist = do_python_eval(f'{args.session_name}/mask_results', args, 0)










