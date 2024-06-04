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

    double *Q,*R;
    qr_factorisation(block,&Q,&R);

    for (int k = 0; k < nprocs; k++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == k) {
            printf("(rank %d)\n", myid);
            printf("R matrix:\n");
            for (int i = 0; i < block.n; i++) {
                for (int j = 0; j < block.n; j++) {
                    printf("%.2f ", R[i * block.n + j]);
                }
                printf("\n");
            }

            printf("Q matrix:\n");
            for (int i = 0; i < block.local_rows; i++) {
                for (int j = 0; j < block.n; j++) {
                    printf("%.2f ", Q[i * block.n + j]);
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

