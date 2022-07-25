#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --mem=200G
#SBATCH --account=ibex-cs

source ~/miniconda3/bin/activate /ibex/ai/home/shaima0d/conda_environ/torch_1_10
srun -c $SLURM_CPUS_PER_TASK python -m torch.distributed.launch --nproc_per_node=4 train_ddp.py
#srun -c $SLURM_CPUS_PER_TASK python  mnist-distributed.py --nodes 1 --gpus 4 --nr 0 --epochs 20
