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

    double *local_vec;
    double t1,t2;
    t1 = MPI_Wtime();
    mpk(n, block, v, myid, nprocs, &local_vec);
    t2 = MPI_Wtime();
    
    if(myid == 0)
    {
      printf("Time taken: %f",t2-t1);
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
    //free(my_vec_pos);
    if (myid == 0)
    {
        free(v);
    }
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
    MPI_Request request;
    MPI_Status status;
    int m = n / nprocs;                           // num components to be distributed to each proc

    if (n % nprocs != 0)
    {
        fprintf(stderr, "Number of processes does not evenly divide matrix size\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    *local_vec = (double *)malloc(m * sizeof(double));
    MPI_Scatter(v, m, MPI_DOUBLE, *local_vec, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // printf("(rank: %d) owns indexes\n",rank);
    // for(int i = 0;i<m;i++)
    // {
    //     printf("%d",(*my_vec_pos)[i]);
    //     printf("\n");
    // }
    int recv_proc;
    int recv_count = 0;
    int *req_indices = NULL;
    bool *index_requested = (bool *)calloc(n, sizeof(bool)); //will track duplicates

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

    // test if identification stage was succesful
    // printf("(rank : %d) required indices\n", rank);
    // for (int i = 0; i < recv_count; i++)
    // {

    //     printf("%d\n", req_indices[i]);
    // }

    // We use a RMA operation to tell the data-sender process that it will be receiving a request to send data
    int is_receiver = 0; // receiver meaning it is a 'request' receiver
    MPI_Win win;
    MPI_Win_create(&is_receiver, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    int req_pos;
    int source_rank;
    double *recv_indices = (double *)malloc(recv_count * sizeof(double));

    double *local_result = (double *)calloc(m, sizeof(double));

    if (recv_count > 0)
    {
        for (int i = 0; i < recv_count; i++)
        {
            // determine who we want to request data from
            int global_index = req_indices[i];
            int proc = (global_index / m) % nprocs;
            // set is_receiver equal to one on the process we want to receive data from
            int value = 1;
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, proc, 0, win);
            MPI_Put(&value, 1, MPI_INT, proc, 0, 1, MPI_INT, win);
            MPI_Win_unlock(proc, win);

            // send the request for data
            MPI_Isend(&req_indices[i], 1, MPI_INT, proc, proc + 100, MPI_COMM_WORLD, &request);

            // DO LOCAL COMPUTATION
            if (local_count > 0)
            {
                for (int i = 0; i < block.local_rows; i++)
                {
                    local_result[i] = 0;
                    for (int j = 0; j < block.cols; j++)
                    {
                        for (int k = 0; k < local_count; k++)
                        {
                            if (j == local_indices[k])
                            {
                                int local_index = j % m;
                                local_result[i] += block.local_data[i * block.cols + j] * (*local_vec)[local_index];
                            }
                        }
                        // local_result[i] += block.local_data[i * block.cols + j] * (*local_vec)[j];
                    }
                }
            }
            // MPI_Wait(&request,&status);
        }
    }
    else
    {
        // DO LOCAL COMPUTATION
        if (local_count > 0)
        {
            for (int i = 0; i < block.local_rows; i++)
            {
                local_result[i] = 0;
                for (int j = 0; j < block.cols; j++)
                {
                    for (int k = 0; k < local_count; k++)
                    {
                        if (j == local_indices[k])
                        {
                            int local_index = j % m;
                            local_result[i] += block.local_data[i * block.cols + j] * (*local_vec)[local_index];
                        }
                    }
                }
            }
        }
    }
    MPI_Win_fence(0, win);
   
    // receive the send request and send the data
    if (is_receiver == 1)
    {
        while (true) // process multiple data-send-requests
        {
            MPI_Irecv(&req_pos, 1, MPI_INT, MPI_ANY_SOURCE, rank + 100, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
            source_rank = status.MPI_SOURCE; // determine who sent the request
            int local_index = req_pos % m;   // compute the local index on this process relative to the global index which was requested

            // send the data
            double send_val = (*local_vec)[local_index];
            MPI_Isend(&send_val, 1, MPI_DOUBLE, source_rank, 1, MPI_COMM_WORLD, &request);
            // MPI_Wait(&request,&status);

            // determine if any more requests are coming through
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE, rank + 100, MPI_COMM_WORLD, &flag, &status);
            if (!flag)
            {
                break;
            }
        }
    }

    // receive the data that was requested earlier
    if (recv_count > 0)
    {
        for (int i = 0; i < recv_count; i++)
        {
            int global_index = req_indices[i];
            int proc = global_index / m;

            double rec_val;
            MPI_Irecv(&rec_val, 1, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
            recv_indices[i] = rec_val;
        }
    }

    //perform final communication-dependent computations
    if (recv_count > 0)
    {
        for (int i = 0; i < block.local_rows; i++)
        {
            for (int j = 0; j < block.cols; j++)
            {
                for (int k = 0; k < recv_count; k++)
                {
                    if (j == req_indices[k])
                    {
                        local_result[i] += block.local_data[i * block.cols + j] * recv_indices[k];
                    }
                }
            }
        }
    }

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

    free(recv_indices);
    free(req_indices);
    free(index_requested);
    free(local_index_dupe);
    free(local_result);
    if (rank == 0)
    {
        free(final_result);
    }
    MPI_Win_free(&win);
}