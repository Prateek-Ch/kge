#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=kge
#SBATCH --partition=accelerated-h100
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
module load devel/cuda/12.4
source ~/miniconda3/bin/activate newkge
export PYTHONPATH=/home/hk-project-test-p0021631/st_st190139/miniconda3/envs/newkge/lib/python3.9/site-packages:$PYTHONPATH

cd kge/util/distribution-shift/
chmod +x train-complex.sh
chmod +x train-fb15k-distmult.sh
chmod +x train-fb15k-complex.sh
bash train-complex.sh
bash train-fb15k-distmult.sh
bash train-fb15k-complex.sh