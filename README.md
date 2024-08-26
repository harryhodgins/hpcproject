# HPC Project 2024

This repository contains the various work I have completed for my master's thesis submitted as part of the degree M.Sc. in High Performance Computing at Trinity College Dublin. 

## Prerequisites

- Access to the TCHPC cluster 'seagull'

## Environment Setup
On seagull, clone the repository using
```bash
git clone https://github.com/harryhodgins/hpcproject

```

The following modules must be loaded on seagull:

```bash
module load openmpi/4.1.6-gcc-8.5.0-fjgly4p tbb compiler-rt intel_ipp_intel64/latest mkl/2024.1
```

## Compilation
The files are compiled by typing `make` in the terminal.

## File Structure
The folder `catsqr` contains files which implement the TSQR algorithm.
The folder `mpk` contains files which implement the matrix powers kernels algorithms.


## Execution
Details on executing the different files are included in the README files of the individual folders. In general, an executable file called `exec` is executed on `x` processes by running the command
```bash
mpirun -n x ./exec
```    

## Acknowledgement
The seagull cluster used for this program is managed and maintained by Research IT.
Information is available at the link below:  

https://www.tchpc.tcd.ie/resources/acknowledgementpolicy
