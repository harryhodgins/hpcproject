#!/bin/sh
#SBATCH -n 64  # cores
#SBATCH -t 1-03:00:00   # 1 day and 3 hours
#SBATCH -p compute      # partition name
#SBATCH -J tsqr  # sensible name for the job
#SBATCH --output=./slurm_files/tsqr_64_%j.out  # Output file name, %j will be replaced by the job ID


# [load up the correct modules, if required](#load-up-the-correct-modules-if-required)
module load openmpi/4.1.6-gcc-8.5.0-fjgly4p intel_ipp_intel64/latest mkl/2024.1

cd /home/users/mschpc/2023/hodginsh/hpcproject/catsqr


# [launch the code](#launch-the-code)
#/home/support/apps/intel/rhel7/19.0.5/compilers_and_libraries_2019.5.281/linux/mpi/intel64/bin/mpirun -n 16 ./poiss2d 15
#mpirun -np $SLURM_NTASKS ./pa1
mpirun ./tsqr
