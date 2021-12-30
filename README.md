### 环境配置
```shell
conda create -n hrnet python=3.7 -y
conda activate hrnet

# torch 1.6
conda install pytorch=1.6.0 torchvision cudatoolkit=10.2 -c pytorch -y
# or torch 1.1
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch -y

cd HRNet-Lesion
pip install -r requirements.txt -i https://pypi.douban.com/simple/
pip install opencv-python -i https://pypi.douban.com/simple
pip install sklearn -i https://pypi.douban.com/simple/
```

### 训练及测试
```shell
# generate lst
python tools/generate_lst.py ../data/IDRID/

# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_lesion.py --cfg experiments/_idrid_/seg_hrnet_w48_idrid.yaml

# resume train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_lesion.py --cfg experiments/_idrid_/seg_hrnet_w48_idrid.yaml --resume-from /path/to/epoch_xxx.pth

# test
# single gpu
CUDA_VISIBLE_DEVICES=0 python tools/test_lesion.py --cfg experiments/_idrid_/seg_hrnet_w48_idrid.yaml --checkpoint /path/to/epoch_xxx.pth

# multiple gpu
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/test_lesion.py --cfg experiments/_idrid_/seg_hrnet_w48_idrid.yaml --checkpoint /path/to/epoch_xxx.pth

```