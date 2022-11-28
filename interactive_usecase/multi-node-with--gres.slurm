#!/bin/bash

#SBATCH --gres=gpu:4
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=200G
#SBATCH --account=ibex-cs


source ~/miniconda3/bin/activate /ibex/ai/home/shaima0d/conda_environ/torch_1_10

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"

# Get the IP address and set port for MASTER node
head_node="${nodes_array[0]}"
echo "Getting the IP address of the head node ${head_node}"
master_ip=$(srun -n 1 -N 1  -w ${head_node} /bin/hostname -I | cut -d " " -f 2)
master_port=10121
echo "head node is ${master_ip}:${master_port}"


for (( i=0; i< ${SLURM_NNODES}; i++ ))
do
     srun -n 1 -N 1 -c $SLURM_CPUS_PER_TASK -w ${nodes_array[i]}  \
      python -m torch.distributed.launch \
     --nproc_per_node=4 --nnodes=2 --node_rank=${i} \
     --master_addr=${master_ip} --master_port=${master_port}\
     train_ddp.py &
 
done
wait
