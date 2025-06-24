#!/bin/bash

#SBATCH -p emmalu
#SBATCH --nodes=1               # Number of nodes
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                # Number of GPUs per node
#SBATCH --mem=64G                      # Memory per node
#SBATCH --time=7-00:00:00 

python twisted_sampler_by_csv.py