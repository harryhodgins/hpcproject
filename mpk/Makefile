# Compiler
GCC = mpicc

# MKL Root directory
MKLROOT = /home/support/rl8/intel/oneapi/2024.1.0/mkl/2024.1

# Compiler flags
GCCFLAGS = -Wall -Wextra -I$(MKLROOT)/include -std=c11 -fopenmp

# Linker flags
LDFLAGS = -L$(MKLROOT)/lib/intel64 \
          -L/home/support/rl8/intel/oneapi/2024.1.0/compiler/2024.1/lib \
          -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
          -liomp5 -lpthread -lm -ldl

# Target
TARGETS = pa0 pa1v2 pa1v1

# Source files
SRCS = pa0.c pa1v2.c pa1v1.c

# Object files
OBJS = $(SRCS:.c=.o)

all: $(TARGETS)

pa1v1: pa1v1.o
	$(GCC) $(GCCFLAGS) -o $@ pa1v1.o $(LDFLAGS)
	
pa0: pa0.o
	$(GCC) $(GCCFLAGS) -o $@ pa0.o $(LDFLAGS)

pa1v2: pa1_v2.o
	$(GCC) $(GCCFLAGS) -o $@ pa1_v2.o $(LDFLAGS)
	
%.o: %.c
	$(GCC) $(GCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) $(OBJS)

.PHONY: all clean
