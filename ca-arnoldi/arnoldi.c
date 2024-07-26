/**
 * @file arnoldi.c
 * @author H.Hodgins (hodginsh@tcd.ie)
 * @brief Implementation of parallel TSQR factorisation using OpenMPI
 * @version 0.1
 * @date 2024-05-30
 * 
 */
#include "arnoldi.h"

int main(int argc, char* argv[])
{
    int myid;
    int nprocs;

    double *full_matrix = NULL;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    //read and distribute the matrix
    MatrixBlock block = distributematrix("arnoldi_test_matrix.txt",myid,nprocs);

    if(myid == 0)
    {
        full_matrix = (double *)malloc(nprocs * block.local_rows * block.n * sizeof(double));
    }
    gather_full_matrix(block.local_A, block.local_rows, block.n, myid, nprocs,full_matrix);

    printf("Process %d received block:\n", myid);
    for(int i = 0; i < block.local_rows; i++)
    {
        for(int j = 0; j < block.n; j++)
        {
            printf("%.1f ", block.local_A[i * block.n + j]);
        }
        printf("\n");
    }

    double *R_final = NULL;
    R_final = (double *)malloc(block.n * block.n * sizeof(double));
    tsqr(block, &R_final, myid, nprocs);
    

    if(myid == 0)
    {
        printf("(rank %d)\n", myid);
        printf("R matrix:\n");
        for(int i = 0; i < block.n; i++)
        {
            for(int j = 0; j < block.n; j++)
            {
                printf("%.2f ", R_final[i * block.n + j]);
            }
            printf("\n");
        }
    }    

    if(myid == 0)
    {
        int n = block.n;
        int lda = block.n;
        int info;

        // Allocate memory for the inverse of R
        double *R_inv = (double *)malloc(n * n * sizeof(double));
        memcpy(R_inv, R_final, n * n * sizeof(double));

        // LAPACK routine to compute the inverse of the upper triangular matrix R
        info = LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'U', 'N', n, R_inv, lda);
        if (info != 0)
        {
            fprintf(stderr, "Error in LAPACKE_dtrtri: %d\n", info);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("R inverse matrix:\n");
        for(int i = 0; i < block.n; i++)
        {
            for(int j = 0; j < block.n; j++)
            {
                printf("%.2f ", R_inv[i * block.n + j]);
            }
            printf("\n");
        }


        double *Q = (double *)malloc(nprocs*block.local_rows * block.n * sizeof(double));

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nprocs*block.local_rows, n, n, 1.0, full_matrix, n, R_inv, n, 0.0, Q, n);

        printf("Q:\n");
        for(int i = 0; i < nprocs*block.local_rows; i++)
        {
            for(int j = 0; j < block.n; j++)
            {
                printf("%.3f ", Q[i * block.n + j]);
            }
            printf("\n");
        }
        free(R_inv);
        free(full_matrix);
    }
    
    free(R_final);

    MPI_Finalize();

    return 0;
}

