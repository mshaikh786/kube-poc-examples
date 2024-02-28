#!/bin/bash

#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH -n 1 
#SBATCH -c 8
#SBATCH -C v100
#SBATCH -t 00:20:0
#SBATCH -A c2227



source /ibex/ai/home/$USER/miniconda3/bin/activate dist-pytorch
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export PYTHONFAULTHANDLER=1



export DATA_DIR=/ibex/ai/reference/CV/tinyimagenet

master_ip=$(/bin/hostname)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 ../ddp.py  --epochs=4 --lr=0.001 --num-workers=${SLURM_CPUS_PER_TASK} --batch-size=32 

#     srun -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} -w ${SLURM_NODELIST} --gpus=${SLURM_GPUS_PER_NODE}  \
#      python -m torch.distributed.launch --use_env \
#     --nproc_per_node=${SLURM_GPUS_PER_NODE} --nnodes=${SLURM_NNODES} --node_rank=${i} \
#     --master_addr=${master_ip} --master_port=${master_port} \
#     ../ddp.py --epochs=10 --lr=0.001 --num-workers=${SLURM_CPUS_PER_TASK} --batch-size=64 

