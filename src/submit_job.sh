#!/bin/bash
#SBATCH -J run
#SBATCH -N 1
#SBATCH -o ./out
#SBATCH -e ./err
#SBATCH -p cpu2

#CUDA_VISIBLE_DEVICES=1
echo "Preprocessing the test data"
./preprocessing_testdata.sh
echo "Contig binning through metabat2"
./metabat_binning.sh
echo "Preprocessing the train data"
./preprocessing_traindata.sh
echo "Start trainning.."
python main.py
echo "Finish!"
