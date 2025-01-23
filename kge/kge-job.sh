#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=kge
#SBATCH --partition=accelerated-h100
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
module load devel/cuda/12.4
source ~/miniconda3/bin/activate newkge
export PYTHONPATH=~/miniconda3/envs/newkge/lib/python3.9/site-packages:$PYTHONPATH

cd kge/util/distribution-shift/
chmod +x train.sh
chmod +x results.sh
bash train.sh
bash results.sh
