#!/bin/bash -l
#SBATCH --job-name="CNN.basic2D"
#SBATCH --output=slurms/220_correction2_%x_%A-%a.out
#SBATCH --error=slurms/220_correction2_%x_%A-%a.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

#SBATCH --array=0-287

export OMP_NUM_THREADS=12
export CRAY_CUDA_MPS=1
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1


module load daint-gpu
module load cray-python/3.6.5.7
module load TensorFlow/1.14.0-CrayGNU-19.10-cuda-10.1.168-python3
module load Horovod/0.16.4-CrayGNU-19.10-tf-1.14.0

srun python3 run.py 	--removed_average 1 \
			--dimensionality 2 \
			--data_location /scratch/snx3000/dprelogo/data/ \
			--saving_location /scratch/snx3000/dprelogo/models/ \
			--logs_location /scratch/snx3000/dprelogo/logs/ \
			--model CNN.basic2D \
			--HyperparameterIndex $SLURM_ARRAY_TASK_ID \
			--epochs 20 \
			--gpus 1 \
