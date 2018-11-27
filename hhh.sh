#!/usr/bin/env bash
source activate detection
# baseline
# mpirun -np 8 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_landmarks.py

# Exp 5: 200 epoch heatmap L2 loss
#mpirun -np 8 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_landmarks.py --tensorboard-path /home/ubuntu/TB5 --save-params-path /home/ubuntu/param,5 --epochs 200 --mixup-epoch 200

# Exp6: 900 epochs < 700 mixup
#mpirun -np 8 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_landmarks.py --tensorboard-path /home/ubuntu/TB6 --save-params-path /home/ubuntu/param,6 --epochs 900 --mixup-epoch 700

# Exp7 BN  wd = 1e-4, grad_clip=0.1, dropout=0.4
#mpirun -np 8 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_landmarks.py --tensorboard-path /home/ubuntu/TB7 --save-params-path /home/ubuntu/param,7 --epochs 900 --mixup-epoch 700

# Exp8 MPL wd = 0
#mpirun -np 8 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_landmarks.py --tensorboard-path /home/ubuntu/TB8 --save-params-path /home/ubuntu/param,8 --epochs 500 --mixup-epoch 400 --lr-epoch 420  --wd 0

# Exp9 Finetune
#mpirun -np 8 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python exp9_finetune.py --tensorboard-path /home/ubuntu/TB9 --save-params-path /home/ubuntu/param,9 --epochs 500 --lr-epoch 300 --data-dir /home/ubuntu/FaceDatasets/WFLW/WFLW_crop_annotations/resplit_0.0_1_sigma_3_64_train.txt --per-batch 28

# Exp10 only train boundary
#mpirun -np 8 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_boundary.py --tensorboard-path /home/ubuntu/TB10 --save-params-path /home/ubuntu/param,10 --epochs 1000 --lr-epoch 800 --data-dir /home/ubuntu/FaceDatasets/WFLW/WFLW_crop_annotations/resplit_0.0_1_sigma_3_64_train.txt --per-batch 28

# Exp11 不小心写错了lr公式
#mpirun -np 8 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_boundary.py --tensorboard-path /home/ubuntu/TB11 --save-params-path /home/ubuntu/param,11 --epochs 1500 --lr-epoch 1000 --mixup-epoch 500 --data-dir /home/ubuntu/FaceDatasets/WFLW/WFLW_crop_annotations/resplit_0.0_1_sigma_3_64_train.txt --per-batch 28 --norm-type GN

# Exp12  一直担心mixup这个东西放在哪里最好：是开头、中间、还是结尾，
#mpirun -np 8 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_boundary.py --tensorboard-path /home/ubuntu/TB12 --save-params-path /home/ubuntu/param,12 --epochs 1500 --lr-epoch 1000 --mixup-epoch 500 --data-dir /home/ubuntu/FaceDatasets/WFLW/WFLW_crop_annotations/resplit_0.0_1_sigma_3_64_train.txt --per-batch 28 --norm-type GN --pretrained 11 --wd 0

# Exp13 Discriminator
mpirun -np 8 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_boundary.py --tensorboard-path /home/ubuntu/TB13 --save-params-path /home/ubuntu/param,13 --epochs 600 --lr-epoch 500 --mixup-epoch 0 --data-dir /home/ubuntu/FaceDatasets/WFLW/WFLW_crop_annotations/resplit_0.0_1_sigma_3_64_train.txt --per-batch 28 --norm-type GN --pretrained 12 --wd 0 --loss-type 1

# Exp14
#mpirun -np 8 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_boundary.py --tensorboard-path /home/ubuntu/TB13 --save-params-path /home/ubuntu/param,13 --epochs 600 --lr-epoch 500 --mixup-epoch 0 --data-dir /home/ubuntu/FaceDatasets/WFLW/WFLW_crop_annotations/resplit_0.0_1_sigma_3_64_train.txt --per-batch 28 --norm-type GN --pretrained 12 --wd 0 --loss-type 1
