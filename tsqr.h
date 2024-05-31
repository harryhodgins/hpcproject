#ifndef TSQR_H
#define TSQR_H

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <mkl_types.h>
#include <mkl_cblas.h>
#include <mkl_lapacke.h>

typedef struct {
    double *local_A;
    int local_rows;
    int n;
} MatrixBlock;
#endif // TSQR_H
