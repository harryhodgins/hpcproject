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

MatrixBlock distributematrix(const char *filename,int rank,int nprocs);
void qr_factorisation(MatrixBlock block,double **tau,double **R);
void pairwise_qr(double *R1, double *R2, int n, double **R_new);
void tsqr(MatrixBlock block,double **R_final,int myid,int nprocs);
void get_q(int myid, MatrixBlock block, double *R_final, double *full_matrix, int nprocs, double **Q);
void gather_full_matrix(double *local_A, int local_rows, int n, int rank, int nprocs,double *full_matrix);

void gather_full_matrix(double *local_A, int local_rows, int n, int rank, int nprocs,double *full_matrix)
{

    MPI_Gather(local_A, local_rows * n, MPI_DOUBLE, full_matrix, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Full matrix gathered at root:\n");
        for (int i = 0; i < local_rows * nprocs; i++) {
            for (int j = 0; j < n; j++) {
                printf("%.2f ", full_matrix[i * n + j]);
            }
            printf("\n");
        }
        
    }
}

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

void tsqr(MatrixBlock block,double **R_final,int myid,int nprocs)
{
    // Stage 0 - perform QR factorization on each local block
    double *tau,*R;
    qr_factorisation(block,&tau,&R);

    // Stage 1 - follow binary tree pattern of QR factorizations on parired blocks
    int level = 0;
    while (nprocs >> level > 1)
    {
        if(myid % (2 << level) == 0) // Receviers -myid multiple of 2^(level+1)
        {
            double *R_recv = (double *)malloc(block.n * block.n * sizeof(double));
            MPI_Recv(R_recv, block.n * block.n, MPI_DOUBLE, myid + (1 << level), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("(rank %d) received from rank %d\n",myid,myid + (1 << level));
            double *R_new;
            pairwise_qr(R, R_recv, block.n, &R_new);
            free(R_recv);
            free(R);
            R = R_new;
        }
        else if(myid % (2 << level) == (1 << level)) // Senders - myid odd multiple of 2^level
        {
            MPI_Send(R, block.n * block.n, MPI_DOUBLE, myid - (1 << level), 0, MPI_COMM_WORLD);
            //printf("(rank %d) sent to rank %d\n",myid,myid - (1 << level));
        }
        MPI_Barrier(MPI_COMM_WORLD);
        level++;
    }

    if(myid == 0)
    {
        *R_final = R;
    }
    else
    {
        free(R);
    }
    free(block.local_A);
    free(tau);
}

/**
 * @brief Get the Q factor of the QR factorisation by computing Q = AR^{-1}
 * 
 * @param myid 
 * @param block 
 * @param R_final 
 * @param full_matrix 
 * @param nprocs 
 * @param Q 
 */
void get_q(int myid, MatrixBlock block, double *R_final, double *full_matrix, int nprocs, double **Q) {
    if (myid == 0) {
        int n = block.n;
        int lda = block.n;
        int info;

        // Allocate memory for the inverse of R
        double *R_inv = (double *)malloc(n * n * sizeof(double));
        memcpy(R_inv, R_final, n * n * sizeof(double));

        // LAPACK routine to compute the inverse of the upper triangular matrix R
        info = LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'U', 'N', n, R_inv, lda);
        if (info != 0) {
            fprintf(stderr, "Error in LAPACKE_dtrtri: %d\n", info);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("R inverse matrix:\n");
        for (int i = 0; i < block.n; i++) {
            for (int j = 0; j < block.n; j++) {
                printf("%.2f ", R_inv[i * block.n + j]);
            }
            printf("\n");
        }

        *Q = (double *)malloc(nprocs * block.local_rows * block.n * sizeof(double));

        // Compute Q = A * R_inv
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nprocs * block.local_rows, n, n, 1.0, full_matrix, n, R_inv, n, 0.0, *Q, n);

        printf("Q matrix:\n");
        for (int i = 0; i < nprocs * block.local_rows; i++) {
            for (int j = 0; j < block.n; j++) {
                printf("%.3f ", (*Q)[i * block.n + j]);
            }
            printf("\n");
        }

        free(R_inv);
        free(full_matrix);
    }
}
   
#endif // TSQR_H