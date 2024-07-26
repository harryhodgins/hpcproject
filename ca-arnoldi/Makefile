# Compiler
GCC = mpicc

# MKL Root directory
MKLROOT = /opt/intel/oneapi/mkl/2024.2

# Compiler flags
GCCFLAGS = -Wall -Wextra -I$(MKLROOT)/include -std=c11
LDFLAGS = -L$(MKLROOT)/lib/intel64 -lmkl_sequential -lmkl_core -lpthread -ldl -llapacke -llapack -lblas

# Target
TARGETS = arnoldi mpk

# Source files
SRCS = arnoldi.c mpk.c

# Object files
OBJS = $(SRCS:.c=.o)

all: $(TARGETS)

arnoldi: arnoldi.o
	$(GCC) $(GCCFLAGS) -o $@ arnoldi.o $(LDFLAGS)

mpk: mpk.o
	$(GCC) $(GCCFLAGS) -o $@ mpk.o $(LDFLAGS)

%.o: %.c
	$(GCC) $(GCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) $(OBJS)

.PHONY: all clean
