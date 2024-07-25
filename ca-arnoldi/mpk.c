#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <mkl_types.h>
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <stdbool.h>


typedef struct {
    double *local_data;
    int local_rows;
    int cols;
} MatrixBlock;

MatrixBlock distributematrix(const char *filename,int rank,int nprocs);
void mpk(int n,MatrixBlock block,double *v,int rank,int nprocs,double **local_vec,int **my_vec_pos);

int main(int argc, char* argv[])
{
    int myid;
    int nprocs;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    //read and distribute the matrix
    MatrixBlock block = distributematrix("mpk_test.txt",myid,nprocs);

    int n = 8;
    double *v = NULL;
    int *my_vec_pos;
    if(myid == 0)
    {
        v = (double *)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            v[i] = i + 1;
        }
    }

    double *local_vec;
    mpk(n,block,v,myid,nprocs,&local_vec,&my_vec_pos);

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
    free(my_vec_pos);
    if(myid == 0)
    {
        free(v);
    }
    MPI_Finalize();
    return 0;
    
}

MatrixBlock distributematrix(const char *filename,int rank,int nprocs)
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

    // broadcast number of cols & rows
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&rows,1,MPI_INT,0,MPI_COMM_WORLD);

    m = rows/nprocs;

    block.local_rows = m;
    block.cols = n;
    block.local_data = (double *)malloc(m*n*sizeof(double));

    //printf("(rank %d), block.local_rows = %d, block.n = %d\n",rank,block.local_rows,block.n);

    // scatter matrix to all procs
    MPI_Scatter(matrix,m*n,MPI_DOUBLE,block.local_data,m*n,MPI_DOUBLE,0,MPI_COMM_WORLD);

    if(rank == 0)
    {
        free(matrix);
    }

    return block;
}

void mpk(int n,MatrixBlock block,double *v,int rank,int nprocs,double **local_vec,int **my_vec_pos)
{
    MPI_Request request;
    MPI_Status status;
    int m = n/nprocs; //num components to be distributed to each proc
    *my_vec_pos = (int *)malloc(m*sizeof(int)); //will hold indexes of v held by each proc

    if(n%nprocs!=0)
    {
        fprintf(stderr,"Number of processes does not evenly divide matrix size\n");
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    *local_vec = (double *)malloc(m * sizeof(double));
    MPI_Scatter(v,m,MPI_DOUBLE,*local_vec,m,MPI_DOUBLE,0,MPI_COMM_WORLD);

    int start_index = rank*m;
    for(int i = 0;i<m;i++)
    {
        (*my_vec_pos)[i] = start_index + i;
    }

    // printf("(rank: %d) owns indexes\n",rank);
    // for(int i = 0;i<m;i++)
    // {
    //     printf("%d",(*my_vec_pos)[i]);
    //     printf("\n");
    // } 
    int recv_proc;
    int recv_count = 0;
    int *req_indices = NULL;
    bool *index_requested = (bool *)calloc(n, sizeof(bool));  // Bitmap to track requested indices

    for(int i = 0;i<block.local_rows;i++)
    {
        for(int j = 0;j<block.cols;j++)
        {
            if(block.local_data[i *block.cols + j] != 0) //identify a non-zero column
            {
                //I need j-th component of v
                recv_proc = j / m; //Identify which proc has this component

                if(rank != recv_proc && !index_requested[j]) //j is not in my local_vec
                {
                    //save index of required component
                    req_indices = (int *)realloc(req_indices,(recv_count+1)*sizeof(int)); //dynamically allocate memory
                    req_indices[recv_count] = j;
                    recv_count++;
                    index_requested[j] = true;
                }
            }
        }
    }

    //test if identification stage was succesful
    // for(int i = 0;i<recv_count;i++)
    // {
    //     printf("(rank : %d)\n",rank);
    //     printf("%d\n",req_indices[i]);
    // }
    
    //We use a RMA operation to tell the data-sender process that it will be receiving a request to send data
    int is_receiver = 0; //receiver meaning it is a 'request' receiver
    MPI_Win win;
    MPI_Win_create(&is_receiver, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    int req_pos; 
    int source_rank;
    int *recv_indices = (int *)malloc(recv_count*sizeof(int));

    if(recv_count >0)
    {
        for(int i = 0;i<recv_count;i++)
        {
            //determine who we want to request data from
            int global_index = req_indices[i]; 
            int proc =  global_index / m;

            //set is_receiver equal to one on the process we want to receive data from
            int value = 1;
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, proc, 0, win);
            MPI_Put(&value, 1, MPI_INT, proc, 0, 1, MPI_INT, win);
            MPI_Win_unlock(proc, win);

            //send the request for data
            MPI_Isend(&req_indices[i],1,MPI_INT,proc,proc+100,MPI_COMM_WORLD,&request);

            //DO LOCAL COMPUTATION

            //MPI_Wait(&request,&status);
            printf("(rank : %d) sent request to rank %d for index %d\n",rank,proc,req_indices[i]);
        }
    }
    // else
    // {
    //     //DO LOCAL COMPUTATION
    // }
    MPI_Win_fence(0, win);

    //receive the send request and send the data
    if(is_receiver == 1) 
    {
        while(true) //process multiple data-send-requests
        {
            MPI_Irecv(&req_pos,1,MPI_INT,MPI_ANY_SOURCE,rank+100,MPI_COMM_WORLD,&request);
            MPI_Wait(&request,&status);
            source_rank = status.MPI_SOURCE; //determine who sent the request
            int local_index = req_pos%m; //compute the local index on this process relative to the global idex which was requested
            //printf("(rank : %d) received request from rank %d for index %d (value = %f)\n",rank,source_rank,req_pos,(*local_vec)[local_index]);

            //send the data
            double send_val = (*local_vec)[local_index];
            MPI_Isend(&send_val, 1, MPI_DOUBLE, source_rank, 1, MPI_COMM_WORLD, &request);
            //MPI_Wait(&request,&status);
            printf("(rank: %d)sent %f (index %d) to rank %d\n",rank,send_val,req_pos,source_rank);

            //determine if any more requests are coming through
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE, rank + 100, MPI_COMM_WORLD, &flag, &status);
            if(!flag)
            {
                break;
            }
        }        
    }


    //receive the data that was requested earlier
    if(recv_count >0)
    {
        for(int i = 0;i<recv_count;i++)
        {
            int global_index = req_indices[i];
            int proc =  global_index / m;

            double rec_val;
            MPI_Irecv(&rec_val,1,MPI_DOUBLE,proc,1,MPI_COMM_WORLD,&request);
            MPI_Wait(&request,&status);
            recv_indices[i] = rec_val;
        }
    }
    
    if(recv_count>0)
    {
        printf("(rank:%d)\n",rank);
        for(int i = 0;i<recv_count;i++)
        {
            printf("%d\n",recv_indices[i]);
        }
    }
    
    free(recv_indices);
    free(req_indices);
    free(index_requested);
    MPI_Win_free(&win);

}