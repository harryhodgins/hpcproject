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
#SBATCH -J pa0_tests_$n  # job name with core count
#SBATCH --output=./slurm_files/pa0_${n}_%j.out  # Output file name

# Load necessary modules
module load openmpi/4.1.6-gcc-8.5.0-fjgly4p tbb compiler-rt intel_ipp_intel64/latest mkl/2024.1

# Change directory

EOF

    # Add the appropriate mpirun command based on the core count
    if [ "$n" -gt 16 ]; then
        echo "mpirun ./pa0" >> "job_${n}.sh"
    else
        echo "mpirun -np $n ./pa0" >> "job_${n}.sh"
    fi

    # Submit the job
    sbatch "job_${n}.sh"
done
