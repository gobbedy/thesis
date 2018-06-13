#!/bin/bash
##SBATCH --cpus-per-task=4 # number of cores
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --time=00:30:00
#SBATCH --job-name="quick test"
#SBATCH --mem=2gb

export PYTHONUNBUFFERED=1
portfolio_simulation.py $@
