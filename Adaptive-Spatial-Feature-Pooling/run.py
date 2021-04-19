import os

batch_size = 128
COEFF = 12
lr = 4e-5
# for i in range(6):
#
#       session_name = f'Lip_pool_cf={COEFF}_bn={batch_size}_lr={lr}_merge_cat'
#       cmd = f'python train.py --session_name={session_name} --batch_size={batch_size} --gpu_index=0 --gpu_num=1 --COEFF={COEFF} --lr={lr}  '
#       os.system(cmd)
#       COEFF+=1
# lrs = [3e-5,4e-5]
# for lr in lrs:
session_name = f'PB_net50_cf=12_3'
cmd = f'python train.py --session_name={session_name} --batch_size={batch_size} --gpu_index=2 --gpu_num=1 --COEFF_1={25} ' \
                        f' --COEFF_2={18} --COEFF_3={13} '
os.system(cmd)
# cmd = f'python train.py --session_name={session_name} --batch_size={batch_size} --gpu_index=2,0,1 --gpu_num=2 --COEFF={COEFF} --lr={lr} --num_epochs={num_epochs} '
num_epochs = 8
# weight_decay = 0
#
# for COEFF in range(5,25):
#
#     session_name = f'PB_net50_cf={COEFF}_lr={lr}_3'
#     cmd = f'python train_2.py --session_name={session_name} --batch_size={batch_size} --gpu_index=2 --gpu_num=1 --COEFF={COEFF} ' \
#         f'  --num_epochs={num_epochs} --lr={lr} '
#     os.system(cmd)



weight_path = '/users4/mxtuo/zhanghan/DBnet/PB_net50_cf=12_lr=0.0001_3/model_weights/res50-bs=128-cf=cf=12.0-47.087.pth'

session_name = 'baseline_res50_bs=128_cf=12_3'
cam_path = f'{session_name}/cam_dir'
crf_dir = f'{session_name}/crf_dir'
pred_dir = f'{session_name}/pred_dir'
voc12_root = '/users4/mxtuo/zhanghan/data/VOC2012'
cmd = f'python infer_cls_ser.py --weights={weight_path} --infer_list=voc12/val.txt --out_cam={cam_path} ' \
      f' --out_crf={crf_dir} --out_cam_pred={pred_dir} --voc12_root={voc12_root} '

# os.system(cmd)
weights = '/users4/mxtuo/zhanghan/model_weight/ilsvrc-cls_rna-a1_cls1000_ep-0001.params'
la_crf_dir = f'/users4/mxtuo/zhanghan/DBnet/Lip_pool_cf=17.5_bn=128_init_baseline_1_3_5_7/crf_dir_4.0'
ha_crf_dir = f'/users4/mxtuo/zhanghan/DBnet/Lip_pool_cf=17.5_bn=128_init_baseline_1_3_5_7/crf_dir_24.0'
# cmd = f'python train_aff.py --weights={weights} --voc12_root={voc12_root} --la_crf_dir={la_crf_dir} ' \
#       f'--ha_crf_dir={ha_crf_dir} --session_name=affnet'
# os.system(cmd)

your_weights_file = f'affnet-ep=2.pth'
your_rw_dir = 'rw_dir_6'
cmd = f'python infer_aff.py --weights={your_weights_file} --infer_list=voc12/val.txt --cam_dir={cam_path} --voc12_root={voc12_root} --out_rw={your_rw_dir}'
# os.system(cmd)

