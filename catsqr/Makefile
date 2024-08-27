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
TARGET = tsqr

# Source files
SRCS = tsqr.c

# Object files
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(GCC) $(GCCFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c
	$(GCC) $(GCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
