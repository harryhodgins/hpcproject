/**
 * @file pa0.c
 * @author H.Hodgins (hodginsh@tcd.ie)
 * @brief Implementation of the PA0 matrix-powers-kernel algorithm from Demmel et al.
 * @version 1.0
 * @date 2024-08-24
 *
 */

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <mkl_types.h>
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <stdbool.h>
#include <omp.h>

typedef struct
{
    double *local_data;
    int local_rows;
    int cols;
} MatrixBlock;

MatrixBlock distributematrix(const char *filename, int rank, int nprocs);
void mpk(int n, MatrixBlock block, double *v, int rank, int nprocs, double *result);

int main(int argc, char *argv[])
{
    int myid;
    int nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // read and distribute the matrix
    MatrixBlock block = distributematrix("delaunay_n12_4096.txt", myid, nprocs);

    int n = 4096;
    double *v = (double *)malloc(n * sizeof(double));
    ;
    if (myid == 0)
    {
        // v = (double *)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++)
        {
            v[i] = i + 1;
        }
    }

    double *result = (double *)calloc(n, sizeof(double));
    double t1, t2, elapsed_time;
    double max_time, min_time, avg_time;
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    mpk(n, block, v, myid, nprocs, result);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
    t2 = MPI_Wtime();

    elapsed_time = t2 - t1;

    // Gather timing information from all processes
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        avg_time /= nprocs;
        printf("Max time: %f seconds\n", max_time);
        printf("Min time: %f seconds\n", min_time);
        printf("Avg time: %f seconds\n", avg_time);
    }

    // printf("Process %d received block of vector:\n", myid);
    // for(int i = 0; i < block.local_rows; i++)
    // {
    //     printf("%.1f ", local_vec[i]);
    //     printf("\n");
    // }

    // printf("Process %d received matrix block:\n", myid);
    // for(int i = 0; i < block.local_rows; i++)
    // {
    //     for(int j = 0; j < block.cols; j++)
    //     {
    //         printf("%.1f ", block.local_data[i * block.cols + j]);
    //     }
    //     printf("\n");
    // }

    free(result);
    free(v);
    MPI_Finalize();
    return 0;
}

MatrixBlock distributematrix(const char *filename, int rank, int nprocs)
{
    double *matrix = NULL;
    int m, n, rows;

    MatrixBlock block;

    if (rank == 0)
    {
        FILE *file = fopen(filename, "r");

        if (file == NULL)
        {
            fprintf(stderr, "Error opening file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fscanf(file, "%d %d", &rows, &n);
        m = rows / nprocs;

        if (rows % nprocs != 0)
        {
            fprintf(stderr, "Number of processes does not evenly divide matrix size\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        matrix = (double *)malloc(rows * n * sizeof(double));

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < n; j++)
            {
                fscanf(file, "%lf", &matrix[i * n + j]);
            }
        }

        fclose(file);
    }

    // broadcast number of cols & rows
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    m = rows / nprocs;

    block.local_rows = m;
    block.cols = n;
    block.local_data = (double *)malloc(m * n * sizeof(double));

    // printf("(rank %d), block.local_rows = %d, block.n = %d\n",rank,block.local_rows,block.n);

    // scatter matrix to all procs
    MPI_Scatter(matrix, m * n, MPI_DOUBLE, block.local_data, m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        free(matrix);
    }

    return block;
}

/**
 * @brief Calculates the Matrix-Vector Product using collective MPI operations.
 * 
 * @param n Size of the square input matrix
 * @param block Matrix block owned by each process.
 * @param v Input vector.
 * @param rank Process ID.
 * @param nprocs Total number of processes in MPI environment.
 * @param result Final result of the computation.
 */
void mpk(int n, MatrixBlock block, double *v, int rank, int nprocs,double *result)
{
    int m = n / nprocs; //num local vector components in each proc

    double *local_result = (double *)calloc(m, sizeof(double));
    MPI_Bcast(v, n, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Broadcast the Vector

    //perform local matrix-vector multiplication
    for(int i = 0;i<m;i++)
    {
        for(int j = 0;j<n;j++)
        {
            local_result[i]+=v[j]*block.local_data[i*block.cols+j];
        }
    }

    //gather results to root
    MPI_Gather(local_result,m,MPI_DOUBLE,result,m,MPI_DOUBLE,0,MPI_COMM_WORLD);

    free(local_result);
    // if (rank == 0)
    // {
    //     for (int i = 0; i < n; i++)
    //     {
    //         printf("%f\n",result[i]);
    //     }
    // }
}