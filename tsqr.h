#ifndef TSQR_H
#define TSQR_H

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <mkl_types.h>
#include <mkl_cblas.h>
#include <mkl_lapacke.h>

typedef struct {
    double *local_A;
    int local_rows;
    int n;
} MatrixBlock;

MatrixBlock distributematrix(const char *filename,int rank,int nprocs)
{
    double *matrix = NULL;
    int m,n,rows;

    MatrixBlock block;

    if(rank == 0)
    {
        FILE *file = fopen(filename,"r");

        if(file == NULL)
        {
            fprintf(stderr,"Error opening file\n");
            MPI_Abort(MPI_COMM_WORLD,1);
        }

        fscanf(file,"%d %d",&rows,&n);
        m = rows/nprocs;

        if(rows%nprocs!=0)
        {
            fprintf(stderr,"Number of processes does not evenly divide matrix size\n");
            MPI_Abort(MPI_COMM_WORLD,1);
        }

        matrix = (double *)malloc(rows*n*sizeof(double));
        
        for(int i = 0;i<rows;i++)
        {
            for(int j = 0;j<n;j++)
            {
                fscanf(file,"%lf",&matrix[i*n+j]);
            }
        }

        fclose(file);
    }

    // broadcast number of cols & rows
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&rows,1,MPI_INT,0,MPI_COMM_WORLD);

    m = rows/nprocs;

    block.local_rows = m;
    block.n = n;
    block.local_A = (double *)malloc(m*n*sizeof(double));

    //printf("(rank %d), block.local_rows = %d, block.n = %d\n",rank,block.local_rows,block.n);

    // scatter matrix to all procs
    MPI_Scatter(matrix,m*n,MPI_DOUBLE,block.local_A,m*n,MPI_DOUBLE,0,MPI_COMM_WORLD);

    if(rank == 0)
    {
        free(matrix);
    }

    return block;
}

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/**
 * @brief Computes the QR factorization of the local block matrix.
 * 
 * @param block The local block matrix.
 * @param tau Matrix holding the householder reflectors used for constructing the Q factor.
 * @param R The R factor of the decomposition.
 */
void qr_factorisation(MatrixBlock block,double **tau,double **R)
{
    *tau = (double *)malloc(block.n*block.n*sizeof(double));

    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, block.local_rows, block.n, block.local_A, block.n, *tau);

    *R = (double *)malloc(block.n*block.n*sizeof(double));
    
    for(int i = 0;i<block.n;i++)
    {
        for(int j = 0;j<block.n;j++)
        {
            if(i<=j)
            {
                (*R)[i*block.n+j] = block.local_A[i*block.n+j]; 
            }
            else
            {
                (*R)[i*block.n+j] = 0;
            }
        }
    }

    // *Q = (double *)malloc(block.local_rows*block.n*sizeof(double));

    // for(int i = 0;i<block.local_rows;i++)
    // {
    //     for(int j = 0;j<block.n;j++)
    //     {
    //         if(i==j)
    //         {
    //             (*Q)[i*block.n+j] = 1.0;
    //         }
    //         else
    //         {
    //             (*Q)[i*block.n+j] = 0;
    //         }
    //     }
    // }

    // LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'N', block.local_rows, block.n,MIN(block.local_rows,block.n), block.local_A, block.n, tau, *Q, block.n);

    // free(tau);
}

/**
 * @brief Computes the QR factorization of the matrix formed by the stacked matrices R1 R2 from a given pairing in the binary tree.
 * 
 * @param R1 
 * @param R2 
 * @param n 
 * @param R_new 
 */
void pairwise_qr(double *R1, double *R2, int n, double **R_new)
{
    double *A = (double *)malloc(2 * n * n * sizeof(double));

    //copy R1 into 'top half' of A
    memcpy(A, R1, n * n * sizeof(double));

    //copy R2 into 'bottom half' of A
    memcpy(A + n * n, R2, n * n * sizeof(double));

    double *tau = (double *)malloc(n * sizeof(double));

    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, 2 * n, n, A, n, tau);

    *R_new = (double *)malloc(n * n * sizeof(double));

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(i <= j)
            {
                (*R_new)[i * n + j] = A[i * n + j];
            } 
            else
            {
                (*R_new)[i * n + j] = 0;
            }
        }
    }

    free(A);
    free(tau);
}
#endif // TSQR_H
