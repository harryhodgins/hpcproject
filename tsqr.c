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
    // double *QR = (double *)malloc(block.local_rows*block.n*sizeof(double));
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, block.local_rows, block.n, block.n, 1.0, Q, block.n, R, block.n, 0.0, QR, block.n);

    // for (int k = 0; k < nprocs; k++)
    // {
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     if(myid == k)
    //     {
    //         printf("(rank %d)\n", myid);
    //         printf("QR matrix:\n");
    //         for(int i = 0; i < block.local_rows; i++)
    //         {
    //             for(int j = 0; j < block.n; j++)
    //             {
    //                 printf("%.2f ", QR[i * block.n + j]);
    //             }
    //             printf("\n");
    //         }
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
    
        if(myid == 0)
        {
            printf("(rank %d)\n", myid);
            printf("R matrix:\n");
            for(int i = 0; i < block.n; i++)
            {
                for(int j = 0; j < block.n; j++)
                {
                    printf("%.2f ", R[i * block.n + j]);
                }
                printf("\n");
            }
        }
    
    free(block.local_A);
    free(tau);
    free(R);
    MPI_Finalize();

    return 0;
}

