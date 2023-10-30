#!/bin/bash
#SBATCH --array=0-1
#SBATCH --job-name=SN_feat
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=45G
#SBATCH -account=ibex-cs

date

echo "Loading anaconda..."
# module purge
module load anaconda3
module load cuda/10.1.243
module list
source activate SoccerNet
echo "...Anaconda env loaded"


echo "Extracting features..."
python tools/ExtractFeatures.py \
--soccernet_dirpath /ibex/scratch/giancos/SoccerNet/ \
--game_ID $SLURM_ARRAY_TASK_ID \
--back_end=TF2 \
--features=ResNET \
--video LQ \
--transform crop \
--verbose \
"$@"

echo "Features extracted..."

date
