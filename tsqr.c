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


    return 0;
}

void distributematrix(const char *filename,int rank,int nprocs)
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
}