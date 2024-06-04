/**
 * @file tsqr.c
 * @author H.Hodgins (hodginsh@tcd.ie)
 * @brief Implementation of parallel TSQR factorisation using OpenMPI
 * @version 0.1
 * @date 2024-05-30
 * 
 */
#include "tsqr.h"

int main(int argc, char* argv[])
{
    int myid;
    int nprocs;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Status status;

    //read and distribute the matrix
    MatrixBlock block = distributematrix("tsqr_test_matrix.txt",myid,nprocs);

    printf("Process %d received block:\n", myid);
    for(int i = 0; i < block.local_rows; i++)
    {
        for(int j = 0; j < block.n; j++)
        {
            printf("%.1f ", block.local_A[i * block.n + j]);
        }
        printf("\n");
    }

    //perform local QR factorisation on each block
    double *Q,*R;
    qr_factorisation(block,&Q,&R);

    double *QR = (double *)malloc(block.local_rows*block.n*sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, block.local_rows, block.n, block.n, 1.0, Q, block.n, R, block.n, 0.0, QR, block.n);

    for (int k = 0; k < nprocs; k++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == k)
        {
            printf("(rank %d)\n", myid);
            printf("QR matrix:\n");
            for(int i = 0; i < block.local_rows; i++)
            {
                for(int j = 0; j < block.n; j++)
                {
                    printf("%.2f ", QR[i * block.n + j]);
                }
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    free(block.local_A);
    free(Q);
    free(R);
    MPI_Finalize();

    return 0;
}

