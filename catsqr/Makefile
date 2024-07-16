# Compiler
GCC = mpicc

# MKL Root directory
MKLROOT = /opt/intel/oneapi/mkl/2024.2

# Compiler flags
GCCFLAGS = -Wall -Wextra -I$(MKLROOT)/include -std=c11
LDFLAGS = -L$(MKLROOT)/lib/intel64 -lmkl_sequential -lmkl_core -lpthread -ldl -llapacke -llapack -lblas

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
