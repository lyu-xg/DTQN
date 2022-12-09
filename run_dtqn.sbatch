#!/bin/bash
#SBATCH --job-name MADTQN
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH --gres gpu:1
#SBATCH --time 08:00:00
#SBATCH --mem 60GB
#SBATCH --exclude c[2160,2162-2175,2176-2183],d1008,d1026
#SBATCH --output outputs/slurm.%x.%j.out
#SBATCH --error outputs/slurm.%x.%j.err
#SBATCH --mail-type FAIL
#SBATCH --mail-user lu.xue@northeastern.edu


python run.py $@
