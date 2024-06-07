# HPC Project 2024

This repository contains the various work I have completed for my master's thesis submitted as part of the degree M.Sc. in High Performance Computing at Trinity College Dublin. 

## Prerequisites

- Access to the TCHPC cluster 'chuck'
- OpenMPI

## Environment Setup

On chuck, load the necessary modules with:
```bash
module load openmpi/3.1.5-gnu9.2.0
module load intel/18.0.4/parallel_studio_xe_2018.4.057
```

## Compilation
The files are compiled by running 'make' in the terminal.

## File Structure
The folder `catsqr` contains files which implement CA-TSQR.

## Execution
The tsqr file can be run on x processes using the following command  
```bash
mpirun -n x ./tsqr
```    

## Acknowledgement
The seagull and chuck clusters used for this program are managed and maintained by Research IT.
Information is available at the link below:  

https://www.tchpc.tcd.ie/resources/acknowledgementpolicy
