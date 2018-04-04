
CC = gcc
CUDA = nvcc
CFLAGS = -g -Wall -Werror -std=c99 -m64 -fopenmp

all: MatMultOpenMP MatMultCUDA

MatMultOpenMP: MatMultOpenMP.c 
	$(CC) $(CFLAGS) -o MatMultOpenMP MatMultOpenMP.c -lm 

MatMultCUDA: MatMultCUDA.cu
	$(CUDA) MatMultCUDA.cu -o MatMultCUDA


#
# Clean the src dirctory
#
clean:
	rm -rf *.o
	rm -f MatMultOpenMP
	rm -f MatMultCUDA
	rm -f *~
