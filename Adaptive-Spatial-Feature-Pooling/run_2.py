import os

batch_size = 64
COEFF = 12
lr = 1e-4
session_name = f'rsnet50'
num_epochs = 7
# weight_decay = 0
#
crop_size = 224
drop_rate = 0.5
network = 'resnet101_atrous'


# session_name = f'res101_cf={COEFF}_lr={lr}'
# cmd = f'python train_2.py --session_name={session_name} --batch_size={batch_size} --gpu_index=0 --gpu_num=1 --COEFF={COEFF} ' \
#      f'  --num_epochs={num_epochs} --lr={lr}  '
# os.system(cmd)

session_name = 'baseline_res101'
cam_path = f'{session_name}/cam_dir'
crf_dir = f'{session_name}/crf_dir'
pred_dir = f'{session_name}/pred_dir'
voc12_root = '/input/VOC2012/VOCdevkit/VOC2012'
weight_path = '/root/zh/Adpnet/res50_cf=12_lr=0.0001/model_weights/res50-bs=96-cf=cf=12.0-40.311.pth'
weight_path_101 = '/root/zh/Adpnet/res101_cf=12_lr=0.0001/model_weights/res50-bs=64-cf=cf=12.0-41.252.pth'
# cmd = f'python infer_cls_ser.py --weights={weight_path} --infer_list=voc12/res101val.txt --out_cam={cam_path} ' \
#       f' --out_crf={crf_dir} --out_cam_pred={pred_dir} --voc12_root={voc12_root} '
# os.system(cmd)

cmd = f'python infer_cls_ser.py --weights={weight_path_101} --infer_list=voc12/train_aug_101.txt --out_cam={cam_path} ' \
      f' --out_crf={crf_dir} --out_cam_pred={pred_dir} --voc12_root={voc12_root} '
# os.system(cmd)

gt_dir = 'baseline_res101'
weights = '/root/zh/model_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params'
la_crf_dir = f'{gt_dir}/crf_dir_4.0'
ha_crf_dir = f'{gt_dir}/crf_dir_24.0'
cam_path = f'{gt_dir}/cam_dir'
session_name = 'affnet_res101'
# voc12_root = '/data/cx/TempCX/VOC2012/'
cmd = f'python train_aff.py --weights={weights} --voc12_root={voc12_root} --la_crf_dir={la_crf_dir} ' \
      f'--ha_crf_dir={ha_crf_dir} --session_name={session_name}  '
# os.system(cmd)

for i in range(4,2,-1):
    your_weights_file = f'{session_name}/affnet-ep={i}.pth'
    your_rw_dir = f'{session_name}/rw_dir_{i}'
    cmd = f'python infer_aff.py --weights={your_weights_file} --infer_list=voc12/val.txt --cam_dir={cam_path} --voc12_root={voc12_root} --out_rw={your_rw_dir}'
    os.system(cmd)