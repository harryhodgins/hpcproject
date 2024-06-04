# Compiler
GCC = mpicc

# MKL Root directory
MKLROOT = /home/support/apps/intel/18.0.4/compilers_and_libraries_2018.5.274/linux/mkl

# Compiler flags
GCCFLAGS = -Wall -Wextra -I$(MKLROOT)/include -std=c11
LDFLAGS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -ldl -llapacke -llapack

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
