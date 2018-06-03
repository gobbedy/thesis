#!/bin/bash
##SBATCH --cpus-per-task=4 # number of cores
#SBATCH --time=00:15:00
#SBATCH --job-name="quick test"
#SBATCH --mem=8gb

export PYTHONUNBUFFERED=1
portfolio_simulation.py
