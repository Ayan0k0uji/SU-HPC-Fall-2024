#!/bin/bash
#SBATCH --job-name=myjob_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --time=00:05:00
#SBATCH --open-mode=append
#SBATCH --output=CUDA.out 

module load cuda/8.0
nvcc -g -G -O0 -DGRID_SIZE_X=32 -DGRID_SIZE_Y=32 -DBLOCK_SIZE_X=16 -DBLOCK_SIZE_Y=16 -DN1=1000 -DM1=1000 -DN2=1000 -DM2=1000 -o MatMulGPU.bin MatMulGPU.cu
./MatMulGPU.bin