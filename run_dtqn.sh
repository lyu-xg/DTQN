#!/bin/bash

module load python/3.8.1
module load cuda/11.3
module load gcc/11.1.0
source venv/bin/activate
mkdir -p outputs

project_name=test_find_treasure2
device=cuda
batch_size=32
model=DTQN
env=find_treasure
#env=gv_memory.5x5.yaml

params_layers=(2)

seeds=$(seq 1 5)

cd /home/lu.xue/scratch/DTQN

for layers in ${params_layers[@]}; do
    for seed in ${seeds[@]}; do
        args="
            --project-name $project_name
            --model $model
            --env $env
            --batch $batch_size
            --layers $layers
            --device $device
            --seed $seed
            "
        jid[1]=$(sbatch run_dtqn.sbatch ${args} | tr -dc '0-9')
        for j in {2..5}; do
            jid[$j]=$(sbatch --dependency=afterok:${jid[$((j-1))]} run_dtqn.sbatch ${args} | tr -dc '0-9')
        done
    done
done