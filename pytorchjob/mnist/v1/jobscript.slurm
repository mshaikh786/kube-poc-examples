#!/bin/bash


#SBATCH --gpus-per-node=1
#SBATCH -t 00:02:0
##SBATCH -A c2227



module load singularity

export IMAGE=kubeflow-demo_cuda-samples-11.6.0.sif

singularity run --nv ${IMAGE} deviceQuery
#simpleMultiGPU 

