#!/bin/bash
#SBATCH -J gpucheck
#SBATCH -N 1
#SBATCH -o /home/jli347/src/out
#SBATCH -p gpu1

nvidia-smi
