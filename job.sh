#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-150 -p alvis
#SBATCH --gpus-per-node=T4:4
#SBATCH -t 01:00:00
#SBATCH -J "master_thesis"

srun ./master_thesis.sif python main.py --data-dir "/data" --num-devices 4
