/**
 * @file pa1_v2.c
 * @author H.Hodgins (hodginsh@tcd.ie)
 * @brief PA1 algorithm from Demmel et al. using collective communication
 * @version 1.0
 * @date 2024-08-25
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

typedef struct
{
    double *local_data;
    int local_rows;
    int cols;
} MatrixBlock;

MatrixBlock distributematrix(const char *filename, int rank, int nprocs);
void mpk(int n, MatrixBlock block, double *v, int rank, int nprocs, double **local_vec);

int main(int argc, char *argv[])
{
    int myid;
    int nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // read and distribute the matrix
    MatrixBlock block = distributematrix("delaunay_n10_1024.txt", myid, nprocs);

    int n = 1024;
    double *v = NULL;
    if (myid == 0)
    {
        v = (double *)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++)
        {
            v[i] = i + 1;
        }
    }

    double t1, t2, elapsed_time;
    double max_time, min_time, avg_time;
    double *local_vec;
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    mpk(n, block, v, myid, nprocs, &local_vec);
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
        free(v);
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

    free(local_vec);
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

void mpk(int n, MatrixBlock block, double *v, int rank, int nprocs, double **local_vec)
{
    int m = n / nprocs; // num components to be distributed to each proc

    if (n % nprocs != 0)
    {
        fprintf(stderr, "Number of processes does not evenly divide matrix size\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    *local_vec = (double *)malloc(m * sizeof(double));
    MPI_Scatter(v, m, MPI_DOUBLE, *local_vec, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int recv_proc;
    int recv_count = 0;
    int *req_indices = NULL;
    bool *index_requested = (bool *)calloc(n, sizeof(bool)); // will track duplicates

    int *local_indices = NULL;
    int local_count = 0;
    bool *local_index_dupe = (bool *)calloc(n, sizeof(bool));

    for (int i = 0; i < block.local_rows; i++)
    {
        for (int j = 0; j < block.cols; j++)
        {
            if (block.local_data[i * block.cols + j] != 0) // identify a non-zero column
            {
                // I need j-th component of v
                recv_proc = (j / m) % nprocs; // Identify which proc has this component

                if (rank != recv_proc && !index_requested[j]) // j is not in my local_vec
                {
                    // save index of required component
                    req_indices = (int *)realloc(req_indices, (recv_count + 1) * sizeof(int)); // dynamically allocate memory
                    req_indices[recv_count] = j;
                    recv_count++;
                    index_requested[j] = true;
                }
                else if (rank == recv_proc && !local_index_dupe[j]) // record locally available
                {
                    local_indices = (int *)realloc(local_indices, (local_count + 1) * sizeof(int));
                    local_indices[local_count] = j;
                    local_count++;
                    local_index_dupe[j] = true;
                }
            }
        }
    }

    int *send_counts = (int *)calloc(nprocs, sizeof(int));
    int *recv_counts = (int *)calloc(nprocs, sizeof(int));

    // Determine send counts for each process
    for (int i = 0; i < recv_count; i++)
    {
        int proc = (req_indices[i] / m) % nprocs;
        send_counts[proc]++;
    }

    double *recv_vals = (double *)malloc(recv_count * sizeof(double));
    // int total_recvs = 0;
    // for (int i = 0; i < nprocs; i++)
    // {
    //     total_recvs += send_counts[i];
    // }

    MPI_Request *recv_requests = (MPI_Request *)malloc(recv_count * sizeof(MPI_Request));
    int recv_index = 0;

    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    // for (int i = 0; i < nprocs; i++)
    // {
    //     printf("(rank: %d) I will be sending %d components to rank %d\n",rank,recv_counts[i],i);
    // }

    int *sdispls = (int *)calloc(nprocs, sizeof(int));
    int *rdispls = (int *)calloc(nprocs, sizeof(int));
    int total_recv_count = 0;

    // calculate displacements to find where data belonging to each proc is
    for (int i = 0; i < nprocs; i++)
    {
        if (i > 0)
        {
            sdispls[i] = sdispls[i - 1] + send_counts[i - 1];
            rdispls[i] = rdispls[i - 1] + recv_counts[i - 1];
        }
        total_recv_count += recv_counts[i];
    }

    int *send_data = (int *)malloc(recv_count * sizeof(int));
    int *recv_data = (int *)malloc(total_recv_count * sizeof(int));

    int *temp_counts = (int *)calloc(nprocs, sizeof(int));
    for (int i = 0; i < recv_count; i++)
    {
        int proc = (req_indices[i] / m) % nprocs;
        send_data[sdispls[proc] + temp_counts[proc]++] = req_indices[i];
    }

    free(temp_counts);

    MPI_Alltoallv(send_data, send_counts, sdispls, MPI_INT, recv_data, recv_counts, rdispls, MPI_INT, MPI_COMM_WORLD);

    int total_sends = 0;
    for (int i = 0; i < nprocs; i++)
    {
        total_sends += recv_counts[i];
    }

    // Start non-blocking receives
    for (int j = 0; j < nprocs; j++)
    {
        for (int i = 0; i < send_counts[j]; i++)
        {
            int global_index = send_data[sdispls[j] + i];
            int target_position = -1;

            // Find the correct position in recv_vals for the global_index
            for (int k = 0; k < recv_count; k++)
            {
                if (req_indices[k] == global_index)
                {
                    target_position = k;
                    break;
                }
            }

            if (target_position != -1) // error checking
            {
                MPI_Irecv(&recv_vals[target_position], 1, MPI_DOUBLE, j, global_index, MPI_COMM_WORLD, &recv_requests[recv_index++]);
            }
        }
    }

    // non-blocking sends
    MPI_Request *send_requests = (MPI_Request *)malloc(total_sends * sizeof(MPI_Request));
    int request_index = 0;
    for (int i = 0; i < nprocs; i++)
    {
        for (int j = 0; j < recv_counts[i]; j++)
        {
            int global_index = recv_data[rdispls[i] + j];
            int local_index = global_index % m;
            double send_val = (*local_vec)[local_index];
            MPI_Isend(&send_val, 1, MPI_DOUBLE, i, global_index, MPI_COMM_WORLD, &send_requests[request_index++]);
        }
    }

    // local computation
    double *local_result = (double *)calloc(m, sizeof(double));

    for (int i = 0; i < block.local_rows; i++)
    {
        local_result[i] = 0;
        for (int j = 0; j < block.local_rows; j++)
        {
            local_result[i] += block.local_data[i * block.cols + rank * m + j] * (*local_vec)[j];
        }
    }

    MPI_Waitall(recv_count, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(total_sends, send_requests, MPI_STATUSES_IGNORE);

    // hasmap style lookup table
    double *lookup_table = (double *)calloc(n, sizeof(double));
    for (int k = 0; k < recv_count; k++)
    {
        lookup_table[req_indices[k]] = recv_vals[k];
    }

    // final communication-dependent computations
    for (int i = 0; i < block.local_rows; i++)
    {
        for (int j = rank * m; j < block.cols; j++)
        {
            if (lookup_table[j] != 0)
            {
                local_result[i] += block.local_data[i * block.cols + j] * lookup_table[j];
            }
        }
    }
    free(lookup_table);

    // gather final result to root
    double *final_result = NULL;
    if (rank == 0)
    {
        final_result = (double *)malloc(n * sizeof(double));
    }

    MPI_Gather(local_result, m, MPI_DOUBLE, final_result, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // if (rank == 0)
    // {
    //     for (int i = 0; i < n; i++)
    //     {
    //         printf("%f\n", final_result[i]);
    //     }
    // }
    free(send_counts);
    free(recv_counts);
    free(sdispls);
    free(rdispls);
    free(send_data);
    free(recv_data);
    free(index_requested);
    free(local_index_dupe);
}