
import torch
import torch.optim as optim
from network import PB_atrous,PB_resnet
from tool.dataset import VOC_Dataset
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
from tool.utils import save_cam, save_mask
import numpy as np
from evaluation import do_python_eval
from tqdm import tqdm
import yaml

voc_root = '/input/VOC2012/VOCdevkit/VOC2012'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--gpu_index', default='1', type=str)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--num_epochs', default=7, type=int)
    parser.add_argument('--session_name', required=True, type=str)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument('--COEFF', default=12.0, type=float)
    # parser.add_argument('--drop', action='store_true', default=False)
    parser.add_argument('--drop_rate', type=float, default=0.5)


    args = parser.parse_args()
    print(args)
    import os

    os.makedirs(args.session_name, exist_ok=True)
    os.makedirs(f'{args.session_name}/model_weights', exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index

    train_dataset = VOC_Dataset(args.train_list, voc_root, transform=transforms.Compose([
        # transforms.RandomResizedCrop(model_ft.input_size),
        transforms.RandomCrop(args.crop_size, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True, drop_last=True)
    val_dataset = VOC_Dataset(args.val_list, voc_root, transform=transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    # loglist = do_python_eval(f'baseline_3_5/pred_dir', args, 0)

    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False,
                                drop_last=True)

    test_dataset = VOC_Dataset(args.val_list, voc_root, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    test_dataloader = DataLoader(test_dataset, shuffle=False, pin_memory=True)

    model = PB_resnet.net( args.COEFF).cuda()
    # model.load_state_dict(torch.load("/users4/mxtuo/zhanghan/DBnet/Lip_pool_cf=17.5_bn=128_init_baseline/model_weights/res50-bs=128-cf=17.5-mIoU=47.140.pth"))
    if (args.gpu_num > 1):
        model = nn.DataParallel(model)
    params = filter(lambda x: x.requires_grad, model.parameters())

    # optim = optim.Adam(params,args.lr,weight_decay=2e-4)
    optim = optim.Adam(params, args.lr)
    criterion = nn.MultiLabelSoftMarginLoss()
    best_mAP = 0.0
    best_mIoU = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        for batch, pack in enumerate(train_dataloader):
            img = pack[1].cuda()
            label = pack[2].cuda()
            _, output = model(img)
            output = output.view_as(label)

            loss = criterion(output, label)

            optim.zero_grad()
            loss.backward()
            optim.step()
            print(f'epoch: {epoch+1}/{args.num_epochs} batch: {batch}/{len(train_dataloader)} batch_loss: {loss:.3f}  ')
        print(f'epoch: {epoch+1}/{args.num_epochs} epoch_loss: {loss:.3f} ')

        model.eval()
        groud_truth = []
        pred_scores = []

        for name, img, label in val_dataloader:
            img = img.cuda()
            label = label.cuda()
            with torch.set_grad_enabled(False):
                cam, output = model(img)
                output = output.view_as(label)

            scores = torch.sigmoid(output).cpu()
            pred_scores.append(scores)
            groud_truth.append(label.cpu())

            b, c, h, w = cam.shape
            cam = F.relu(cam)
            cam_max = torch.max(cam.view(b, c, -1), dim=-1)[0].view(b, c, 1, 1) + 1e-5
            cam = F.relu(cam - 1e-5, inplace=True) / cam_max
            cam = cam * (label.view(b, c, 1, 1))

            save_cam(img[0], label[0], cam[0], scores[0], name[0], epoch, args)

        pred_scores = torch.cat(tuple(pred_scores))
        groud_truth = torch.cat(tuple(groud_truth))
        mAP = average_precision_score(groud_truth, pred_scores)
        f = open(f'{args.session_name}/results_mAP.txt', 'a')
        print(f'epoch: {epoch}/{args.num_epochs}  mAP: {mAP:.3f} ')
        print(f'epoch: {epoch}/{args.num_epochs}  mAP: {mAP:.3f}', file=f)
        f.close()
        if (mAP > best_mAP):
            best_mAP = mAP

        if (mAP > 0.845):
            for name, img, label in tqdm(test_dataloader):
                img = img.cuda()
                with torch.set_grad_enabled(False):
                    cam, _, = model(img, is_eval=True)
                b, c, h, w = cam.shape
                cam = F.relu(cam).cpu()
                cam_max = torch.max(cam.view(b, c, -1), dim=-1)[0].view(b, c, 1, 1) + 1e-5
                cam = F.relu(cam - 1e-5, inplace=True) / cam_max
                cam = cam * (label.view(b, c, 1, 1))
                cam = F.interpolate(cam, img.shape[2:], mode='bilinear', align_corners=False).squeeze()
                bg_socre = (torch.ones(img.shape[2:]) * 0.2).unsqueeze(0)
                pred = torch.argmax(torch.cat((bg_socre, cam), dim=0), dim=0).numpy().astype(np.uint8)
                save_mask(pred, args, name[0])
            loglist = do_python_eval(f'{args.session_name}/mask_results', args, epoch)
            mIoU = loglist['mIoU']

            if (best_mIoU < mIoU):
                best_mIoU = mIoU
                torch.save(model.state_dict(),
                           f'/root/zh/Adpnet/{args.session_name}/model_weights/res50-bs={args.batch_size}-cf=cf={args.COEFF}-{mIoU:.3f}.pth')






