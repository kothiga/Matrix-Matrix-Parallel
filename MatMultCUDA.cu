/*
* CPSC 4210
*  - High Performance Parallel Computing
*
*    Name: Austin Kothig
*      ID: 001182645
*     Sem: Spring 2018
*
* Purpose:
*
*
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <iostream>


/* Enable / Disable debugging */
#define debug 0

/* Thread Count */
#define BLOCK 16

/* For running all Matrix Matrix Multiplication Tests */
void RunAllTests (int n, int t);

/* Helper Function Prototypes */
float randomize   (int *seed);
void  clear       (int n, float *X);
void  stats       (char* desc, int n, int threads, double *T, double *R);
void  help        ( );
void  getGPUStats (cudaDeviceProp& prop);
int   validate    (int n, float *S, float *X);

/* Matrix Multiplication Prototypes*/
void global_cuda (int n, int t, float *A, float *B, float *C);
void shared_cuda (int n, int t, float *A, float *B, float *C);



/* Kernal Function Implementation */
__global__
void global_cuda_kernal(int n, float* A, float* B, float* C) {

  //-- get current position
  const unsigned int ROW = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int COL = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.f;

  //-- make sure we are valid
  if (ROW < n && COL < n) {

    //-- compute the element in block
    for (int i = 0; i < n; i++) {
      sum += B[ROW*n + i] * C[i*n + COL];
    }
  }

  //-- write sum to device memory
  A[ROW*n + COL] = sum;
}

__global__
void shared_cuda_kernal(int n, float* A, float* B, float* C) {

  //-- get the current for each core
  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  const unsigned int I = blockIdx.y * BLOCK + ty;
  const unsigned int J = blockIdx.x * BLOCK + tx;

  const unsigned int gy = gridDim.y;

  //-- allocate shared memory on the device
  __shared__ float d_b[BLOCK][BLOCK], d_c[BLOCK][BLOCK];

  //-- check that we are in range
  if (I < n && J < n) {

    float sum = 0.f;

    //-- scan through the elements in grid
    for (int i = 0; i < gy; i++) {

      //-- load the block from device memory to shared memory
      d_b[ty][tx] = B[I*n + i*BLOCK + tx];
      d_c[ty][tx] = C[J+n*(i*BLOCK + ty)];

      //-- wait for all threads to load device memory
      //-- into shared memory before continuing.
      __syncthreads();


      //-- multiply the shared memories together
      for (int j = 0; j < BLOCK; j++) {
        sum += d_b[ty][j] * d_c[j][tx];
      }

      //-- wait for all calculations to finish
      __syncthreads();
    }

    //-- write to device memory
    A[I*n + J] = sum;
  }
}



#if debug

/* Used to build a validation Matrix */
void optim_serial (int n, float *A, float *B, float *C);

/* Variables for error checking */
int ErrorCount = 0;
float *s;

#endif


/* Global Variables */
cudaEvent_t time_begin;
cudaEvent_t time_stop;

double avgTime_Global;  double avgRate_Global;
double avgTime_Shared;  double avgRate_Shared;



//--
//--  Main
//--
int main (int argc, char *argv[]) {

  //--
  //-- @@@ SH Note 1b:
  //--  These values need to be read in from command line.
  int n = -1;
  int t = -1;

  //-- loop through arguments
  int opt;
  while ((opt = getopt(argc, argv, "hn:t:")) != -1) {
    switch (opt) {
      case 'h': help(); exit(0); break;
      case 'n': n = atoi(optarg); break;
      case 't': t = atoi(optarg); break;
      default :
      printf("wrong argument\n");
      exit(0); break;
    }
  }


  //-- check to see if we missed any arguments
  if (n == -1) {
    printf("\n\n./MatMultCUDA: Missing required n!!\n");
    help();
    return 0;
  } if (t == -1) {
    // -- make t max if not specified
    //TODO: t = omp_get_max_threads();
  }


  //-- display general information
  printf ( "\n" );
  printf ( "Dense NxN\n" );
  printf ( "  CUDA version.\n" );
  printf ( "\n" );
  printf ( "  Matrix multiplication tests.\n" );


  #if debug
  //--
  //-- generate a validation matrix, and give debug stats
  //--
  printf("n is %d\n", n);
  printf("t is %d\n", t);

  int i, j;
  float* b = (float *) malloc (n*n*sizeof (float));
  float* c = (float *) malloc (n*n*sizeof (float));

  //--
  //-- Assign randomly generated values to the input matrices B and C.
  //--
  int seed = 123456789;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      b[i*n + j] = randomize (&seed);
    }
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      c[i*n + j] = randomize (&seed);
    }
  }

  //-- allocate the space for s
  s = (float *) malloc (n*n*sizeof (float));


  //-- Generate a "Good" Solution
  optim_serial (n, s, b, c);
  printf("\n\nFinished Generating Solution Mat.\n\n");

  free(b); free(c);
  #endif

  //-- Display FOPS
  unsigned long long ops;
  ops  = (unsigned long long)n;
  ops *= (unsigned long long)n;
  ops *= (unsigned long long)n;
  ops *= 2;
  printf("  Floating point OPS roughly = %llu\n", ops);


  //--
  //-- @@@ SH Note 1a:
  //--  You must read in the dimension of the matrix and the number of threads
  //--  from the command line.
  //-- cuda initializations
  cudaDeviceProp prop;
  getGPUStats(prop);

  printf ( "\n" );
  printf ( "  Thread Blocks = %d\n", (((n+BLOCK-1)/BLOCK)*((n+BLOCK-1)/BLOCK))-((n+BLOCK-1)/BLOCK));
  printf ( "  Threads Per Block %d\n", BLOCK*BLOCK);


  avgTime_Global = 0.0;  avgRate_Global = 0.0;
  avgTime_Shared = 0.0;  avgRate_Shared = 0.0;


  for (int i = 1; i <= 10; i++) {
    printf("\n\n\n\n   Beginning Trial %d, of Matrix Size %d\n", i, n);
    printf(        "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
    //-- call the matrix multiplication routines for serial cases
    RunAllTests(n, t);
  }

  avgTime_Global /= 10.0;  avgRate_Global /= 10.0;
  avgTime_Shared /= 10.0;  avgRate_Shared /= 10.0;


  printf("\n\n\n   Total Averages for All 10 CUDA Trials   \n");
  printf(      "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
  printf("  Global Time %f\n  Global Rate %f\n\n", avgTime_Global, avgRate_Global);
  printf("  Shared Time %f\n  Shared Rate %f\n\n", avgTime_Shared, avgRate_Shared);


  //--
  //-- Terminate.
  //--
  printf("\n");
  printf("Dense NxN:\n");
  printf("  Normal end of execution.\n" );

  #if debug
  printf("  Execution Finished with %d Error(s) Found.\n", ErrorCount);
  //-- Deallocate the used memory
  free(s);
  #endif

  return 0;
}


//--
//-- Run a series of NxN Matrix Matrix multiplication
//--  using different stratagies
//--
void RunAllTests (int n, int t) {

  //--
  //-- Variables used in this function
  //--
  int i; int j; int seed;
  double T; double R;

  //--
  //-- Allocate the storage for matrices.
  //--
  float *a;   float *b;   float *c;
  a = (float *) malloc (n*n*sizeof (float));
  b = (float *) malloc (n*n*sizeof (float));
  c = (float *) malloc (n*n*sizeof (float));


  //--
  //-- Assign randomly generated values to the input matrices B and C.
  //--
  seed = 123456789;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      b[i*n + j] = randomize (&seed);
    }
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      c[i*n + j] = randomize (&seed);
    }
  }

  clear(n, a);



  //######################################################
  //--
  //-- Run the Global CUDA Test
  //--
  //######################################################

  //-- create an event
  cudaEventCreate(&time_begin);
  cudaEventCreate(&time_stop);

  //-- run the test
  global_cuda(n, t, a, b, c);


  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("global_cuda is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char global_cuda_desc[] = "Global CUDA.";
  stats(global_cuda_desc, n, t, &T, &R);

  //-- add to averages
  avgTime_Global += T;
  avgRate_Global += R;

  //-- destroy the cuda events
  cudaEventDestroy(time_begin);
  cudaEventDestroy(time_stop);


  //-- Clear out Mat A
  clear(n, a);





  //######################################################
  //--
  //-- Run the Shared CUDA Test
  //--
  //######################################################

  //-- create an event
  cudaEventCreate(&time_begin);
  cudaEventCreate(&time_stop);

  //-- run the test
  shared_cuda (n, t, a, b, c);


  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("shared_cuda is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char shared_cuda_desc[] = "Shared CUDA.";
  stats(shared_cuda_desc, n, t, &T, &R);

  avgTime_Shared += T;
  avgRate_Shared += R;

  //-- destroy the cuda events
  cudaEventDestroy(time_begin);
  cudaEventDestroy(time_stop);

  //-- Clear out Mat A
  clear(n, a);


  //-- Deallocate the used memory
  free(a);   free(b);   free(c);

  return;
}


//--
//-- Get a randomized value, and refresh seed.
//--
float randomize (int *seed) {
  int k; float r;
  k = *seed / 127773;
  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;
  if ( *seed < 0 ) { *seed = *seed + 2147483647; }
  r = (float) (*seed) * 4.656612875E-10;
  return r;
}


//--
//-- clear out the contents of X
//--
void clear (int n, float *X) {
  int i ,j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      X[i*n + j] = 0.f;
    }
  }
}


//--
//-- compare the passed in matracies to see
//-- if there are any differences between them
//--
int validate (int n, float *S, float *X) {

  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (abs(S[i*n + j] - X[i*n + j]) > 0.001) {

        std::cout << "\n\n\n\n";
        std::cout << "Fail at pos " << i*n << " x " << j << std::endl;
        std::cout << S[i*n + j] << " != " << X[i*n + j] << std::endl;

        return 1;
      }
    }
  }

  return 0;
}


//--
//-- Stats : give the user the stats of this implementation
//--
void stats (char* desc, int n, int thread, double *T, double *R) {

  unsigned long long ops;
  float time;
  double rate;

  ops  = (unsigned long long)n;
  ops *= (unsigned long long)n;
  ops *= (unsigned long long)n;
  ops *= 2;


  cudaEventElapsedTime(&time, time_begin, time_stop);

  time /= 1000.f;

  rate = ( double ) ( ops ) / (time) / 1000000.0;

  printf("\n############################################\n");
  printf("  Test    = %s\n", desc);
  printf("  N       = %d\n", n);
  printf("  Threads = %d\n", thread);
  printf("  Floating point OPS roughly = %llu\n", ops);
  printf("  Elapsed time dT            = %f\n", time);
  printf("  Rate = MegaOPS/dT          = %f\n", rate);

  (*T) = time;
  (*R) = rate;
}


//--
//-- Help : simple function for how to use this program
//--
void help () {
  printf("\n");
  printf("Usage: ./MatMultCUDA [-h] -n <num> -t <num> \n");
  printf("Options:\n");
  printf("  -h\t\tPrint this help message.\n");
  printf("  -n <num>\tSize of N.\n");
  printf("  -t <num>\tNumber of Threads.\n");
  printf("Examples:\n");
  printf("linux> ./MatMultCUDA -n 1024 -t 8\n");
}


//--
//-- getGPUStats : print out general information about the GPU
//--
void getGPUStats (cudaDeviceProp &prop) {

  int count;
  cudaGetDeviceCount(&count);

  for (int i = 0; i < count; i++) {
    cudaGetDeviceProperties(&prop, i);
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "Name                       " << prop.name                     << std::endl;
    std::cout << "GPU clock rate             " << (double)prop.clockRate / 1024 << " MHz" << std::endl;
    std::cout << "Registers Per Block        " << prop.regsPerBlock  << std::endl;
    std::cout << "Compute capability         " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total global memory        " << (double)prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Total constant memory      " << (double)prop.totalConstMem / (1024) << " KB" << std::endl;
    std::cout << "Shared memory per block    " << (double)prop.sharedMemPerBlock / (1024) << " KB" << std::endl;
    std::cout << "Maximum threads per block  " << prop.maxThreadsPerBlock << std::endl << std::endl;
    std::cout << "Maximum threads along   X  " << prop.maxThreadsDim[0] << std::endl;
    std::cout << "                        Y  " << prop.maxThreadsDim[1] << std::endl;
    std::cout << "                        Z  " << prop.maxThreadsDim[2] << std::endl << std::endl;
    std::cout << "Maximum grid size along X  " << prop.maxGridSize[0]   << std::endl;
    std::cout << "                        Y  " << prop.maxGridSize[1]   << std::endl;
    std::cout << "                        Z  " << prop.maxGridSize[2]   << std::endl << std::endl;
    std::cout << "Warp size                  " << prop.warpSize            << std::endl;
    std::cout << "Multiprocessor count       " << prop.multiProcessorCount << std::endl;
    std::cout << "Device overlap             " << prop.deviceOverlap       << std::endl << std::endl;
    std::cout << "Maximum resident threads   " << prop.maxThreadsPerMultiProcessor << std::endl
    << "  per multi-processor  \n";

    std::cout << std::endl;
  }
}


//--
//--  Implementation of Different NxN Matrix Multiplication
//--

//--
//-- global_cuda : use global memory on GPU to multiply two matracies
//--
void global_cuda (int n, int t, float *A, float *B, float *C) {

  //-- initialize variables
  float *d_A; float *d_B; float *d_C;

  //-- Allocate Memory on the GPU
  cudaMalloc(&d_A, n*n*sizeof (float));
  cudaMalloc(&d_B, n*n*sizeof (float));
  cudaMalloc(&d_C, n*n*sizeof (float));

  //-- copy data over to gpu
  cudaMemcpy(d_A, A, n*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, n*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, n*n*sizeof(float), cudaMemcpyHostToDevice);

  //-- initialize blocks and threads per blocks
  dim3 DimBlock(BLOCK, BLOCK);
  dim3 DimGrid((n + DimBlock.x - 1) / DimBlock.x,
               (n + DimBlock.y - 1) / DimBlock.y);
  size_t SharedMemBytes = 128;

  //-- recored when the event begin
  cudaEventRecord(time_begin);


  //-- Start the Kernal
  global_cuda_kernal<<<DimGrid,DimBlock,SharedMemBytes>>>(n, d_A, d_B, d_C);


  //-- sync the threads
  cudaThreadSynchronize();

  //-- record when the event ended
  cudaEventRecord(time_stop);

  //-- sync the events
  cudaEventSynchronize(time_stop);

  //-- copy the results out of gpu
  cudaMemcpy(A, d_A, n*n*sizeof(float), cudaMemcpyDeviceToHost);

  //-- Deallocate device Memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}


//--
//-- shared_cuda : use shared memory on GPU to multiply two matracies
//--
void shared_cuda (int n, int t, float *A, float *B, float *C) {

  //-- initialize variables
  float *d_A; float *d_B; float *d_C;

  //-- Allocate Memory on the GPU
  cudaMalloc(&d_A, n*n*sizeof (float));
  cudaMalloc(&d_B, n*n*sizeof (float));
  cudaMalloc(&d_C, n*n*sizeof (float));

  //-- copy data over to gpu
  cudaMemcpy(d_A, A, n*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, n*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, n*n*sizeof(float), cudaMemcpyHostToDevice);

  //-- initialize blocks and threads per blocks
  dim3 DimBlock(BLOCK, BLOCK);
  dim3 DimGrid((n + DimBlock.x - 1) / DimBlock.x,
               (n + DimBlock.y - 1) / DimBlock.y);
  size_t SharedMemBytes = 128;


  //-- recored when the event begin
  cudaEventRecord(time_begin);


  //-- Start the Kernal
  shared_cuda_kernal<<<DimGrid,DimBlock,SharedMemBytes>>>(n, d_A, d_B, d_C);

  //-- sync the threads
  cudaThreadSynchronize();

  //-- record when the event ended
  cudaEventRecord(time_stop);

  //-- sync the events
  cudaEventSynchronize(time_stop);

  //-- copy the results out of gpu
  cudaMemcpy(A, d_A, n*n*sizeof(float), cudaMemcpyDeviceToHost);

  //-- Deallocate device Memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}


#if debug
//--
//-- optim_serial : kij row by row with fixed B.
//--
//-- notes : good cache performance, serial.
//--         used to build a validation matrix.
//--
void optim_serial (int n, float *A, float *B, float *C) {

  int i, j, k;
  float r;

  for (k = 0; k < n; k++) {
    for (i = 0; i < n; i++) {
      r = B[i*n + k];
      for (j = 0; j < n; j++) {
        A[i*n + j] += r * C[k*n + j];
      }
    }
  }
}
#endif
