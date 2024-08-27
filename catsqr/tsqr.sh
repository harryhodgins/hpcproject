#!/bin/sh
#SBATCH -n 32  # cores
#SBATCH -t 1-03:00:00   # 1 day and 3 hours
#SBATCH -p compute      # partition name
#SBATCH -J tsqr  # sensible name for the job
#SBATCH --output=./slurm_files/tsqr_32_%j.out  # Output file name, %j will be replaced by the job ID


# [load up the correct modules, if required](#load-up-the-correct-modules-if-required)
module load openmpi/4.1.6-gcc-8.5.0-fjgly4p intel_ipp_intel64/latest mkl/2024.1

# [launch the code](#launch-the-code)
#mpirun -np $SLURM_NTASKS ./tsqr
mpirun ./tsqr