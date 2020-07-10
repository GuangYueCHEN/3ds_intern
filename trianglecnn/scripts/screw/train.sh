#!/usr/bin/env bash

## run the training
python E:/tutor/train.py \
--dataroot E:/tutor/datasets/SCREW  --name screw_seg  --arch meshunet  --dataset_mode segmentation  --ncf 32 64 128 256  --ninput_edges 2280  --pool_res 1800 1350 600  --resblocks 3  --batch_size 4  --lr 0.001  --num_aug 20  --slide_verts 0.2 --niter 10 --niter_decay 6