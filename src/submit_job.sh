#!/bin/bash
#SBATCH -J run
#SBATCH -N 1
#SBATCH -o ./out
#SBATCH -e ./err
#SBATCH -p gpu1

#CUDA_VISIBLE_DEVICES=1
python graph_generate.py
