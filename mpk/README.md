# Matrix Powers Kernel

This folder contains C files which contain implementations of the PA0 and PA1 algorithms created by Demmel et al. in 'Avoiding Communication in Sparse Matrix Computations'.

## File Structure

- `pa0.c` contains the implementation of the PA0 algorithm.
- `pa1v1.c` contains the PA1 algorithm implemented using point-to-point communication.
- `pa1_v2.c` contains the PA1 algorithm implemented using collective communication.
- `scaling_tests_x.sh` is a bash script which submits numerous jobs to seagull, utilising a various number of nodes/cores. There are three bash scripts, one for each of the c files.

## Prerequisites

- Access to the TCHPC cluster 'seagull'


The following modules must be loaded on seagull:

```bash
module load openmpi/4.1.6-gcc-8.5.0-fjgly4p tbb compiler-rt intel_ipp_intel64/latest mkl/2024.1
```

## Compilation
The files are compiled by typing `make` in the terminal.


## Execution
- Note: git does not seem to transfer the file permissions for the bash scripts below, so it is necessary to run the `chmod` command first as shown.
- To execute the pa0 code, run the command
```bash
chmod +x scaling_tests.sh
./scaling_tests.sh
```    
- To execute the pa1v1 code, run the command
```bash
chmod +x scaling_tests_pa1v1.sh
./scaling_tests_pa1v1.sh
```    
- To execute the pa1v2 code, run the command
```bash
chmod +x scaling_tests_pa1v2.sh
./scaling_tests_pa1v2.sh
```    

The output will appear in a directory called `slurm_files`.


## Acknowledgement
The seagull cluster used for this program is managed and maintained by Research IT.
Information is available at the link below:  

https://www.tchpc.tcd.ie/resources/acknowledgementpolicy