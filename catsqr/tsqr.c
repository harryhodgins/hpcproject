/**
 * @file tsqr.c
 * @author H.Hodgins (hodginsh@tcd.ie)
 * @brief Implementation of parallel TSQR factorisation using OpenMPI
 * @version 0.2
 * @date 2024-05-30
 * 
 */
#include "tsqr.h"

int main(int argc, char* argv[])
{
    int myid;
    int nprocs;

    double *full_matrix = NULL;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    //read and distribute the matrix
    MatrixBlock block = distributematrix("tsqr_test_matrix.txt",myid,nprocs);

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

    double *Q;
    if(myid == 0)
    {
        get_q(myid,block,R_final,full_matrix,nprocs,&Q);
    }
    
    free(R_final);

    MPI_Finalize();

    return 0;
}

