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
#define debug 1


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
void naive_shared (int n, int t, float *A, float *B, float *C);
void optim_shared (int n, int t, float *A, float *B, float *C);
void block_shared (int n, int t, int b, float *A, float *B, float *C);

void naive_global (int n, int t, float *A, float *B, float *C);
void optim_global (int n, int t, float *A, float *B, float *C);
void block_global (int n, int t, int b, float *A, float *B, float *C);

#if debug
/* Used to build a validation Matrix */
void optim_serial (int n, float *A, float *B, float *C);
#endif

/* Global Variables */
cudaEvent_t time_begin;
cudaEvent_t time_stop;

double avgTime_Naive_Shared;
double avgTime_Optim_Shared;
double avgTime_Block_Shared;

double avgTime_Naive_Global;
double avgTime_Optim_Global;
double avgTime_Block_Global;

double avgRate_Naive_Shared;
double avgRate_Optim_Shared;
double avgRate_Block_Shared;

double avgRate_Naive_Global;
double avgRate_Optim_Global;
double avgRate_Block_Global;


#if debug
int ErrorCount = 0;
float *s;
#endif



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
      case 'h': help(); return 0; break;
      case 'n': n = atoi(optarg); break;
      case 't': t = atoi(optarg); break;
      default :
      printf("wrong argument\n");
      return 0; break;
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
  //TODO: printf ( "  Number of processors available = %d\n", omp_get_num_procs());
  //TODO: printf ( "  Number of threads              = %d\n", t);

  avgTime_Naive_Shared = 0.0;
  avgTime_Optim_Shared = 0.0;
  avgTime_Block_Shared = 0.0;

  avgRate_Naive_Shared = 0.0;
  avgRate_Optim_Shared = 0.0;
  avgRate_Block_Shared = 0.0;

  for (int i = 1; i <= 10; i++) {
    printf("\n\n\n\n   Beginning Trial %d, of Matrix Size %d\n", i, n);
    printf(        "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
    //-- call the matrix multiplication routines for serial cases
    RunAllTests(n, t);
  }

  avgTime_Naive_Shared /= 10.0;
  avgTime_Optim_Shared /= 10.0;
  avgTime_Block_Shared /= 10.0;

  avgRate_Naive_Shared /= 10.0;
  avgRate_Optim_Shared /= 10.0;
  avgRate_Block_Shared /= 10.0;


  printf("\n\n\n   Total Averages for All 10 Serial Trials   \n");
  printf(      "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
  printf("  Naive Time %f\n  Naive Rate %f\n\n", avgTime_Naive_Shared, avgRate_Naive_Shared);
  printf("  Optim Time %f\n  Optim Rate %f\n\n", avgTime_Optim_Shared, avgRate_Optim_Shared);
  printf("  Block Time %f\n  Block Rate %f\n\n", avgTime_Block_Shared, avgRate_Block_Shared);


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
  //-- Allocate Memory on the GPU
  //--
  /*
  float *a_d; float *b_d; float *c_d;
  cudaMalloc(&a_d, n*n*sizeof (float));
  cudaMalloc(&b_d, n*n*sizeof (float));
  cudaMalloc(&c_d, n*n*sizeof (float));
  */

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

  //######################################################
  //--
  //-- Run the Naive Shared Test
  //--
  cudaEventCreate(&time_begin); // create an event
  cudaEventCreate(&time_stop);

  cudaEventRecord(time_begin);  // recored when the event begin

  naive_shared (n, t, a, b, c); // run the test

  cudaEventRecord(time_stop);   // record when the event ended


  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("naive_shared is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char naive_shared_desc[] = "Naive Shared.";
  stats(naive_shared_desc, n, t, &T, &R);

  avgTime_Naive_Shared += T;
  avgRate_Naive_Shared += R;

  //-- Clear out Mat A
  clear(n, a);





  //######################################################
  //--
  //-- Run the Optimized shared Test
  //--
  cudaEventCreate(&time_begin); // create an event
  cudaEventCreate(&time_stop);

  cudaEventRecord(time_begin);  // recored when the event begin

  optim_shared (n, t, a, b, c); // run the test

  cudaEventRecord(time_stop);   // record when the event ended


  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("optim_shared is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char optim_shared_desc[] = "Loop Optimized Shared.";
  stats(optim_shared_desc, n, t, &T, &R);

  avgTime_Optim_Shared += T;
  avgRate_Optim_Shared += R;

  //-- Clear out Mat A
  clear(n, a);





  //######################################################
  //--
  //-- Run a series of Blocking Shared Test
  //--
  cudaEventCreate(&time_begin); // create an event
  cudaEventCreate(&time_stop);

  cudaEventRecord(time_begin);  // recored when the event begin

  block_shared(n, t, 16, a, b, c); // run the test

  cudaEventRecord(time_stop);   // record when the event ended


  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("Blocking Shared is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char blocking_shared_desc[] = "Blocking-16 Shared.";
  stats(blocking_shared_desc, n, t, &T, &R);

  avgTime_Block_Shared += T;
  avgRate_Block_Shared += R;

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
      X[i*n + j] = 0.0;
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
      if (S[i*n + j] != X[i*n + j]) {

        for (i = 0; i < n; i++) {
          for (j = 0; j < n; j++) {
            std::cout << S[i*n + j] << " ";
          } std::cout << "      \t";
          for (j = 0; j < n; j++) {
            std::cout << X[i*n + j] << " ";
          } std::cout << std::endl;
        }


        return 1;
      }
    }
  }
  /*
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      std::cout << S[i*n + j] << " ";
    } std::cout << "      \t";
    for (j = 0; j < n; j++) {
      std::cout << X[i*n + j] << " ";
    } std::cout << std::endl;
  }
  */

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

  rate = ( double ) ( ops ) / (time / 1000) / 1000000.0;

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
//-- naive_shared : simple row by column for fixed A.
//--
//-- Notes : poor cache performance, using shared memory on GPU
//--
void naive_shared (int n, int t, float *A, float *B, float *C) {

  int i, j, k;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        A[i*n + j] = A[i*n + j] + (B[i*n + k] * C[k*n + j]);
      }
    }
  }
}


//--
//-- optim_serial : kij row by row with fixed B.
//--
//-- notes : good cache performance, using shared memory on GPU
//--
void optim_shared (int n, int t, float *A, float *B, float *C) {

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


//--
//-- block_serial : ijk based blocks
//--
//-- notes : compromise of temporal and spatial locality, using shared memory on GPU
//--
void block_shared (int n, int t, int b, float *A, float *B, float *C) {

  int i, j, k, en, jj, kk;
  float sum = 0.0;
  en = b * (n/b);

  for (kk = 0; kk < en; kk += b) {
    for (jj = 0; jj < en; jj += b) {
      for (i = 0; i < n; i++) {
        for (j = jj; j < jj+b; j++) {
          sum = A[i*n + j];
          for (k = kk; k < kk+b; k++) {
            sum += B[i*n + k] * C[k*n + j];
          }
          A[i*n + j] = sum;
        }
      }
    }
  }
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
