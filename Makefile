#compiler
GCC = mpicc

ICC = icc
MKLROOT = /home/support/apps/intel/18.0.4/compilers_and_libraries_2018.5.274/linux/mkl

#Compiler flags
GCCFLAGS = -Wall -Wextra -llapacke -llapack -I$(MKLROOT)/include -L$(MKLROOT)/lib/intel64 \
           -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread  -ldl

all: tsqr

tsqr:
	$(GCC) $(GCCFLAGS) -o tsqr.c -g

clean:
	rm -f tsqr

.PHONY: all clean
