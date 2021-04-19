import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
voc_class = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
             'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']


def save_cam(img,label,cam,score,name,epoch,args):
    os.makedirs(f'{args.session_name}/cam_results',exist_ok=True)
    _,h,w = img.shape
    cam = F.interpolate(torch.unsqueeze(cam,0),(h,w),mode='bilinear',align_corners=False)
    cam = torch.squeeze(cam)
    mean = (0.485,0.456,0.406)
    std = (0.229,0.224,0.225)
    img[0, :, :] = ((img[0,:,:]*std[0])+mean[0])*255
    img[1, :, :] = ((img[1, :, :] * std[1]) + mean[1]) * 255
    img[2, :, :] = ((img[2, :, :] * std[2]) + mean[2]) * 255
    img[img>255] = 255
    img[img<0] = 0
    img = img.cpu().numpy().astype(np.uint8)
    img = cv2.cvtColor(img.transpose((1,2,0)),cv2.COLOR_RGB2BGR)
    for i in range(len(torch.nonzero(label))):
        index = torch.nonzero(label)[i]
        cam_class = torch.squeeze(cam[index])
        heatmap = cv2.applyColorMap(np.uint8(255*cam_class.cpu()), cv2.COLORMAP_JET)
        heatmap  = heatmap * 0.4 + img
        cv2.imwrite(f'./{args.session_name}/cam_results/{name}-epoch-{epoch}-{voc_class[index]}-{score[index].item():.2f}.jpg',heatmap)

def save_mask(mask,args,name):
    os.makedirs(f'{args.session_name}/mask_results', exist_ok=True)
    palette = Image.open('/input/VOC2012/VOCdevkit/VOC2012/SegmentationClass/2011_003271.png').getpalette()
    sudo_mask = Image.fromarray(mask).convert('P')
    sudo_mask.putpalette(palette)
    sudo_mask.save(f'{args.session_name}/mask_results/{name}.png')







