#!/bin/bash

# List of core counts
cores=(1 2 4 8 16 32 64)

# Loop through each core count and submit a job
for n in "${cores[@]}"; do
    # Create a job script for each core count
    cat > "job_${n}.sh" << EOF
#!/bin/sh
#SBATCH -n $n  # number of cores
#SBATCH -t 1-03:00:00  # 1 day and 3 hours
#SBATCH -p compute  # partition name
#SBATCH -J pa1v1_tests_$n  # job name with core count
#SBATCH --output=./slurm_files/pa1v1${n}_%j.out  # Output file name

# Load necessary modules
module load openmpi/4.1.6-gcc-8.5.0-fjgly4p tbb compiler-rt intel_ipp_intel64/latest mkl/2024.1

# Change directory
cd /home/users/mschpc/2023/hodginsh/hpcproject/mpk

EOF

    # Add the appropriate mpirun command based on the core count
    if [ "$n" -gt 16 ]; then
        echo "mpirun ./pa1v1" >> "job_${n}.sh"
    else
        echo "mpirun -np $n ./pa1v1" >> "job_${n}.sh"
    fi

    # Submit the job
    sbatch "job_${n}.sh"
done
