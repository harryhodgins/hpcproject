# TSQR

This folder contains C files which contain a distributed-memory parallel implementation of the TSQR algorithm created by Demmel et al.

## File Structure

- `tsqr.c` contains the main file.
- `tsqr.h` contains the TSQR algorithm and necessary functions.
- `tsqr.sh` runs the code on 32 cores when executes, this value can be edited.

## Prerequisites

- Access to the TCHPC cluster 'seagull'


The following modules must be loaded on seagull:

```bash
module load openmpi/4.1.6-gcc-8.5.0-fjgly4p tbb compiler-rt intel_ipp_intel64/latest mkl/2024.1
```

## Compilation
The files are compiled by typing `make` in the terminal.


## Execution
- To execute the code, run the command
```bash
chmod +x tsqr.sh
sbatch tsqr.sh
```    

The output will appear in a directory called `slurm_files`.

## Acknowledgement
The seagull cluster used for this program is managed and maintained by Research IT.
Information is available at the link below:  

https://www.tchpc.tcd.ie/resources/acknowledgementpolicy