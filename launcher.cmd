#!/bin/bash

#SBATCH --job-name="test_food"

#SBATCH --workdir=.

#SBATCH --output=food_%j.out

#SBATCH --error=food_%j.err

#SBATCH --ntasks=1

#SBATCH --gres gpu:1

#SBATCH --time=01:30:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML/ pydot graphviz

python food11_cnn.py


