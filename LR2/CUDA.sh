#!/bin/bash
#SBATCH --job-name=myjob_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --time=00:05:00
#SBATCH --open-mode=append
#SBATCH --output=CUDA.out 

module load cuda/8.0
nvcc -g -std=c++11 -G -O0 -DN=1000000 -o VectorSum.bin VectorSum.cu
./VectorSum.bin