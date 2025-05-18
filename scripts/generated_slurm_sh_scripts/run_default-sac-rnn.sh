#!/bin/bash
#SBATCH --job-name=dl-default-sac-rnn
#SBATCH --account=nct328
#SBATCH --qos=acc
#SBATCH --time=01-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nct/nct01026/ATCI-P1-BLOBTRACER/
#SBATCH --output=/home/nct/nct01026/ATCI-P1-BLOBTRACER/out_logs/default-sac-rnn/job_output.log
#SBATCH --error=/home/nct/nct01026/ATCI-P1-BLOBTRACER/out_logs/default-sac-rnn/job_error.log

module purge

module load  impi  intel  hdf5  mkl  python/3.12.1-gcc
#module load EB/apps EB/install cuda/12.6 cudnn/9.6.0-cuda12

cd ~/ATCI-P1-BLOBTRACER/
python src/bsc_main.py -c default_sac_rnn
