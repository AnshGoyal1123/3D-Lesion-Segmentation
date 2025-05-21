#!/bin/bash

#SBATCH --account=rsteven1_gpu
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --export=ALL

python /home/agoyal19/Previous_Work/3D-Lesion-Segmentation/GenrativeMethod/model/baseline.py
