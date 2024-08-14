/**
 * @file mpk.c
 * @author H.Hodgins (hodginsh@tcd.ie)
 * @brief An attempt to implement the PA1 matrix-powers-kernel algorithm from Demmel et al.
 * @version 1.0
 * @date 2024-07-29
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
void mpk(int n, MatrixBlock block, double *v, int rank, int nprocs, double **local_vec);

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
    double *v = NULL;
    if (myid == 0)
    {
        v = (double *)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++)
        {
            v[i] = i + 1;
        }
    }

    double *local_vec;
    double t1, t2;
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    mpk(n, block, v, myid, nprocs, &local_vec);
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();

    if (myid == 0)
    {
        printf("Time taken: %f\n", t2 - t1);
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

/**
 * @brief Computes a matrix vector product using the PA1 Matrix Powers Kernel algorithm discussed by Demmel et al.
 *
 * @param n Input matrix dimension.
 * @param block Local matrix block owned by each process.
 * @param v Input vector.
 * @param rank Processor ID.
 * @param nprocs Total number of processors in MPI environment.
 * @param local_vec Local vector block owned by each process.
 * @param my_vec_pos Don't use this?
 */
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

    double *local_mat = (double *)malloc(block.local_rows * block.local_rows * sizeof(double));
    for (int i = 0; i < block.local_rows; i++)
    {
        for (int j = 0; j < m; j++)
        {
            local_mat[i * m + j] = block.local_data[i * block.cols + rank * m + j];
        }
    }

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

    // for (int i = 0; i < nprocs; i++)
    // {
    //     printf("(rank: %d) I need %d indices from rank %d\n", rank, send_counts[i], i);
    // }

    int **send_buffers = (int **)malloc(nprocs * sizeof(int *)); // holds indices each proc will send
    int **recv_buffers = (int **)malloc(nprocs * sizeof(int *)); // holds indices each proc will recieve

    for (int i = 0; i < nprocs; i++)
    {
        if (send_counts[i] > 0)
        {
            send_buffers[i] = (int *)malloc(send_counts[i] * sizeof(int));
        }
    }

    int *temp_counts = (int *)calloc(nprocs, sizeof(int));
    for (int i = 0; i < recv_count; i++)
    {
        int proc = (req_indices[i] / m) % nprocs;
        send_buffers[proc][temp_counts[proc]++] = req_indices[i];
    }

    free(temp_counts);

    double *recv_vals = (double *)malloc(recv_count * sizeof(double));
    // int total_recvs = 0;
    // for (int i = 0; i < nprocs; i++)
    // {
    //     total_recvs += send_counts[i];
    // }

    MPI_Request *recv_requests = (MPI_Request *)malloc(recv_count * sizeof(MPI_Request));
    int recv_index = 0;
    // printf("(rank :%d)\n", rank);
    // for (int j = 0; j < nprocs; j++)
    // {
    //     for (int i = 0; i < send_counts[j]; i++)
    //     {
    //         // printf("I need global index %d from rank %d\n", send_buffers[j][i], j);
    //         MPI_Irecv(&recv_vals[recv_index], 1, MPI_DOUBLE, j, j + send_buffers[j][i], MPI_COMM_WORLD, &recv_requests[recv_index]);
    //         recv_index++;
    //     }
    // }

    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    // for (int i = 0; i < nprocs; i++)
    // {
    //     printf("(rank: %d) I will be sending %d components to rank %d\n",rank,recv_counts[i],i);
    // }

    int *sdispls = (int *)calloc(nprocs, sizeof(int));
    int *rdispls = (int *)calloc(nprocs, sizeof(int));
    int total_recv_count = 0;

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

    temp_counts = (int *)calloc(nprocs, sizeof(int));
    for (int i = 0; i < recv_count; i++)
    {
        int proc = (req_indices[i] / m) % nprocs;
        send_data[sdispls[proc] + temp_counts[proc]++] = req_indices[i];
    }

    free(temp_counts);

    MPI_Alltoallv(send_data, send_counts, sdispls, MPI_INT, recv_data, recv_counts, rdispls, MPI_INT, MPI_COMM_WORLD);
    double *local_result = (double *)calloc(m, sizeof(double));

    int total_sends = 0;
    for (int i = 0; i < nprocs; i++)
    {
        total_sends += recv_counts[i];
    }

    MPI_Request *send_requests = (MPI_Request *)malloc(total_sends * sizeof(MPI_Request));
    int request_index = 0;
    for (int i = 0; i < nprocs; i++)
    {
        for (int j = 0; j < recv_counts[i]; j++)
        {
            int global_index = recv_data[rdispls[i] + j];
            int local_index = global_index % m;
            double send_val = (*local_vec)[local_index];
            // printf("(rank: %d) will be sending index %d (my index %d) to rank %d\n", rank, global_index, local_index, i);
            MPI_Isend(&send_val, 1, MPI_DOUBLE, i,global_index, MPI_COMM_WORLD, &send_requests[request_index++]);
        }
    }
    cblas_dgemv(CblasRowMajor, CblasNoTrans, block.local_rows, block.local_rows, 1.0, local_mat, block.local_rows, *local_vec, 1, 0.0, local_result, 1);

    // for (int j = 0; j < nprocs; j++)
    // {
    //     for (int i = 0; i < send_counts[j]; i++)
    //     {
    //         // printf("I need global index %d from rank %d\n", send_buffers[j][i], j);
    //         MPI_Irecv(&recv_vals[recv_index], 1, MPI_DOUBLE, j,send_buffers[j][i], MPI_COMM_WORLD, &recv_requests[recv_index]);
    //         recv_index++;
    //     }
    // }
    for (int j = 0; j < nprocs; j++)
{
    for (int i = 0; i < send_counts[j]; i++)
    {
        int global_index = send_buffers[j][i];
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

        // Start the non-blocking receive using the global_index as the tag
        if (target_position != -1) // Make sure we have a valid position
        {
            MPI_Irecv(&recv_vals[target_position], 1, MPI_DOUBLE, j, global_index, MPI_COMM_WORLD, &recv_requests[recv_index]);
            recv_index++;
        }
    }
}
    

    // if (rank == 1)
    // {
    //     for (int i = 0; i < m; i++)
    //     {
    //         printf("%f\n", local_result[i]);
    //     }
    // }
    MPI_Waitall(recv_count, recv_requests, MPI_STATUSES_IGNORE);

    //printf("(rank: %d) received vals:\n",rank);
    // for(int i = 0;i<recv_count;i++)
    // {
    //     printf("%f\n",recv_vals[i]);
    // }

    free(send_buffers);
    free(recv_buffers);
    free(send_counts);
    free(recv_counts);
    free(sdispls);
    free(rdispls);
    free(send_data);
    free(recv_data);

    // printf("(rank: %d)\n",rank);
    // for(int i = 0;i<recv_count;i++)
    // {
    //     printf("%f\n",recv_vals[i]);
    // }
    //hash-map style datastructure for the final multiplication
    double *lookup_table = (double *)calloc(n, sizeof(double));
    for (int k = 0; k < recv_count; k++)
    {
        lookup_table[req_indices[k]] = recv_vals[k];
    }

    //final communication-dependent computations
    #pragma omp parallel for
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
    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            printf("%f\n", final_result[i]);
        }
    }

    free(index_requested);
    free(local_index_dupe);
    free(local_result);
     if (rank == 0)
     {
         free(final_result);
     }
}