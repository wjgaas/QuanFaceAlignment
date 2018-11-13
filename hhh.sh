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
mpirun -np 8 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_landmarks.py --tensorboard-path /home/ubuntu/TB8 --save-params-path /home/ubuntu/param,8 --epochs 500 --mixup-epoch 400 --lr-epoch 420  --wd 0