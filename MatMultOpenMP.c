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
#include <omp.h>

/* Enable / Disable debugging */
#define debug 0


/* For running all Matrix Matrix Multiplication Tests */
void RunAllSerialTests (int n);
void RunAllThreadTests (int n, int t);


/* Helper Function Prototypes */
double randomize (int *seed);
void   clear     (int n, double **X);
void   stats     (char* desc, int n, int threads, double *T, double *R);
void   help      ();
int    validate  (int n, double **S, double **X);


/* Matrix Multiplication Prototypes*/
void naive_serial (int n, double **A, double **B, double **C);
void naive_omp_s  (int n, int t, double **A, double **B, double **C);
void naive_omp_d  (int n, int t, double **A, double **B, double **C);
void naive_omp_g  (int n, int t, double **A, double **B, double **C);
void optim_serial (int n, double **A, double **B, double **C);
void optim_omp_s  (int n, int t, double **A, double **B, double **C);
void optim_omp_d  (int n, int t, double **A, double **B, double **C);
void optim_omp_g  (int n, int t, double **A, double **B, double **C);
void block_serial (int n, int b, double **A, double **B, double **C);
void block_omp_s  (int n, int t, int b, double **A, double **B, double **C);
void block_omp_d  (int n, int t, int b, double **A, double **B, double **C);
void block_omp_g  (int n, int t, int b, double **A, double **B, double **C);


/* Global Variables */
double time_begin;
double time_stop;

double avgTime_Naive_Serial;  double avgRate_Naive_Serial;
double avgTime_Optim_Serial;  double avgRate_Optim_Serial;
double avgTime_Block_Serial;  double avgRate_Block_Serial;

double avgTime_Naive_Static;  double avgRate_Naive_Static;
double avgTime_Optim_Static;  double avgRate_Optim_Static;
double avgTime_Block_Static;  double avgRate_Block_Static;

double avgTime_Naive_Dynamic; double avgRate_Naive_Dynamic;
double avgTime_Optim_Dynamic; double avgRate_Optim_Dynamic;
double avgTime_Block_Dynamic; double avgRate_Block_Dynamic;

double avgTime_Naive_Guided;  double avgRate_Naive_Guided;
double avgTime_Optim_Guided;  double avgRate_Optim_Guided;
double avgTime_Block_Guided;  double avgRate_Block_Guided;


#define DYNAMC_CHUNK 128

#if debug
int ErrorCount = 0;
double **s;
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
  int serialFlag = 0;
  int parallelFlag = 0;


  //-- loop through arguments
  int opt;
  while ((opt = getopt(argc, argv, "hn:t:sp")) != -1) {
    switch (opt) {
      case 'h': help(); return 0; break;
      case 'n': n = atoi(optarg); break;
      case 't': t = atoi(optarg); break;
      case 's': serialFlag = 1;    break;
      case 'p': parallelFlag = 1; break;
      default :
      printf("wrong argument\n");
      return 0; break;
    }
  }


  //-- check to see if we missed any arguments
  if (n == -1) {
    printf("\n\n./MatMultOpenMP: Missing required n!!\n");
    help();
    return 0;
  } if (t == -1) {
    // -- make t max if not specified
    t = omp_get_max_threads();
  }


  #if debug
  printf("n is %d\n", n);
  printf("t is %d\n", t);
  #endif


  //-- display general information
  printf ( "\n" );
  printf ( "Dense NxN\n" );
  printf ( "  C/OpenMP version.\n" );
  printf ( "\n" );
  printf ( "  Matrix multiplication tests.\n" );

  unsigned long long ops;

  ops  = (unsigned long long)n;
  ops *= (unsigned long long)n;
  ops *= (unsigned long long)n;
  ops *= 2;
  printf("  Floating point OPS roughly = %llu\n", ops);


  #if debug
  int i, j;

  double** b = (double **) calloc (n, sizeof (double));
  double** c = (double **) calloc (n, sizeof (double));

  for (i = 0; i < n; i++) b[i] = (double *) calloc (n, sizeof (double));
  for (i = 0; i < n; i++) c[i] = (double *) calloc (n, sizeof (double));


  //--
  //-- Assign randomly generated values to the input matrices B and C.
  //--
  int seed = 123456789;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      b[i][j] = randomize (&seed);
    }
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      c[i][j] = randomize (&seed);
    }
  }
  //-- allocate the space for s
  s = (double **) calloc (n, sizeof (double));
  for (i = 0; i < n; i++) s[i] = (double *) calloc (n, sizeof (double));

  //-- Generate a "Good" Solution
  optim_serial (n, s, b, c);
  printf("\n\nFinished Generating Solution Mat.\n\n");

  for (int i = 0; i < n; i++) {
    free(b[i]); free(c[i]);
  }
  free(b); free(c);
  #endif


  //--
  //-- @@@ SH Note 1a:
  //--  You must read in the dimension of the matrix and the number of threads
  //--  from the command line.
  printf ( "\n" );
  printf ( "  Number of processors available = %d\n", omp_get_num_procs());
  printf ( "  Number of threads              = %d\n", t);


  if (serialFlag) {

    avgTime_Naive_Serial = 0.0;  avgRate_Naive_Serial = 0.0;
    avgTime_Optim_Serial = 0.0;  avgRate_Optim_Serial = 0.0;
    avgTime_Block_Serial = 0.0;  avgRate_Block_Serial = 0.0;

    for (int i = 1; i <= 10; i++) {
      printf("\n\n\n\n   Beginning Trial %d, of Matrix Size %d\n", i, n);
      printf(        "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
      //-- call the matrix multiplication routines for serial cases
      RunAllSerialTests(n);
    }

    avgTime_Naive_Serial /= 10.0;  avgRate_Naive_Serial /= 10.0;
    avgTime_Optim_Serial /= 10.0;  avgRate_Optim_Serial /= 10.0;
    avgTime_Block_Serial /= 10.0;  avgRate_Block_Serial /= 10.0;


    printf("\n\n\n   Total Averages for All 10 Serial Trials   \n");
    printf(      "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
    printf("  Naive Time %f\n  Naive Rate %f\n\n", avgTime_Naive_Serial, avgRate_Naive_Serial);
    printf("  Optim Time %f\n  Optim Rate %f\n\n", avgTime_Optim_Serial, avgRate_Optim_Serial);
    printf("  Block Time %f\n  Block Rate %f\n\n", avgTime_Block_Serial, avgRate_Block_Serial);
  }



  if (parallelFlag) {

    avgTime_Naive_Static = 0.0;  avgRate_Naive_Static = 0.0;
    avgTime_Optim_Static = 0.0;  avgRate_Optim_Static = 0.0;
    avgTime_Block_Static = 0.0;  avgRate_Block_Static = 0.0;

    avgTime_Naive_Dynamic = 0.0; avgRate_Naive_Dynamic = 0.0;
    avgTime_Optim_Dynamic = 0.0; avgRate_Optim_Dynamic = 0.0;
    avgTime_Block_Dynamic = 0.0; avgRate_Block_Dynamic = 0.0;

    avgTime_Naive_Guided = 0.0;  avgRate_Naive_Guided = 0.0;
    avgTime_Optim_Guided = 0.0;  avgRate_Optim_Guided = 0.0;
    avgTime_Block_Guided = 0.0;  avgRate_Block_Guided = 0.0;


    for (int i = 1; i <= 10; i++) {
      printf("\n\n\n\n   Beginning Trial %d, of Matrix Size %d, with %d threads\n", i, n, t);
      printf(        "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
      //-- call the matrix multiplication routines for thread cases
      RunAllThreadTests(n, t);
    }

    avgTime_Naive_Static /= 10.0;  avgRate_Naive_Static /= 10.0;
    avgTime_Optim_Static /= 10.0;  avgRate_Optim_Static /= 10.0;
    avgTime_Block_Static /= 10.0;  avgRate_Block_Static /= 10.0;

    avgTime_Naive_Dynamic /= 10.0; avgRate_Naive_Dynamic /= 10.0;
    avgTime_Optim_Dynamic /= 10.0; avgRate_Optim_Dynamic /= 10.0;
    avgTime_Block_Dynamic /= 10.0; avgRate_Block_Dynamic /= 10.0;

    avgTime_Naive_Guided /= 10.0;  avgRate_Naive_Guided /= 10.0;
    avgTime_Optim_Guided /= 10.0;  avgRate_Optim_Guided /= 10.0;
    avgTime_Block_Guided /= 10.0;  avgRate_Block_Guided /= 10.0;


    printf("\n\n\n   Total Averages for All 10 OpenMP Trials   \n");
    printf(      "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
    printf("  Naive-Static  Time %f\n  Naive-Static  Rate %f\n\n",   avgTime_Naive_Static,  avgRate_Naive_Static);
    printf("  Naive-Dynamic Time %f\n  Naive-Dynamic Rate %f\n\n",   avgTime_Naive_Dynamic, avgRate_Naive_Dynamic);
    printf("  Naive-Guided  Time %f\n  Naive-Guided  Rate %f\n\n\n", avgTime_Naive_Guided,  avgRate_Naive_Guided);

    printf("  Optim-Static  Time %f\n  Optim-Static  Rate %f\n\n",   avgTime_Optim_Static,  avgRate_Optim_Static);
    printf("  Optim-Dynamic Time %f\n  Optim-Dynamic Rate %f\n\n",   avgTime_Optim_Dynamic, avgRate_Optim_Dynamic);
    printf("  Optim-Guided  Time %f\n  Optim-Guided  Rate %f\n\n\n", avgTime_Optim_Guided,  avgRate_Optim_Guided);

    printf("  Block-Static  Time %f\n  Block-Static  Rate %f\n\n",   avgTime_Block_Static,  avgRate_Block_Static);
    printf("  Block-Dynamic Time %f\n  Block-Dynamic Rate %f\n\n",   avgTime_Block_Dynamic, avgRate_Block_Dynamic);
    printf("  Block-Guided  Time %f\n  Block-Guided  Rate %f\n\n\n", avgTime_Block_Guided,  avgRate_Block_Guided);

  }


  //--
  //-- Terminate.
  //--
  printf("\n");
  printf("Dense NxN:\n");
  printf("  Normal end of execution.\n" );

  #if debug
  printf("  Execution Finished with %d Error(s) Found.\n", ErrorCount);

  //-- Deallocate the used memory
  for (i = 0; i < n; i++) {
    free(s[i]);
  }
  free(s);
  #endif

  return 0;
}


//--
//-- Run a series of NxN Matrix Matrix multiplication
//--  using different stratagies
//--
void RunAllSerialTests (int n) {

  //--
  //-- Variables used in this function
  //--
  double **a; double **b; double **c;
  int i; int j; int seed;
  double T; double R;

  //--
  //-- Allocate the storage for matrices.
  //--
  a = (double **) calloc (n, sizeof (double));
  b = (double **) calloc (n, sizeof (double));
  c = (double **) calloc (n, sizeof (double));

  for (i = 0; i < n; i++) a[i] = (double *) calloc (n, sizeof (double));
  for (i = 0; i < n; i++) b[i] = (double *) calloc (n, sizeof (double));
  for (i = 0; i < n; i++) c[i] = (double *) calloc (n, sizeof (double));


  //--
  //-- Assign randomly generated values to the input matrices B and C.
  //--
  seed = 123456789;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      b[i][j] = randomize (&seed);
    }
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      c[i][j] = randomize (&seed);
    }
  }

  //######################################################
  //--
  //-- Run the Naive Serial Test
  //--
  //######################################################
  naive_serial (n, a, b, c);

  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("naive_serial is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char naive_serial_desc[] = "Naive Serial.";
  stats(naive_serial_desc, n, 0, &T, &R);

  avgTime_Naive_Serial += T;
  avgRate_Naive_Serial += R;

  //-- Clear out Mat A
  clear(n, a);





  //######################################################
  //--
  //-- Run the Optimized Serial Test
  //--
  //######################################################
  optim_serial (n, a, b, c);

  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("optim_serial is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char optim_serial_desc[] = "Loop Optimized Serial.";
  stats(optim_serial_desc, n, 0, &T, &R);

  avgTime_Optim_Serial += T;
  avgRate_Optim_Serial += R;

  //-- Clear out Mat A
  clear(n, a);





  //######################################################
  //--
  //-- Run a series of Serial Blocking Tests
  //--
  //######################################################
  block_serial(n, 16, a, b, c);

  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("Blocking Serial is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char blocking_serial_desc[] = "Serial Blocking - 16.";
  stats(blocking_serial_desc, n, 0, &T, &R);

  avgTime_Block_Serial += T;
  avgRate_Block_Serial += R;

  //-- Clear out Mat A
  clear(n, a);

  //-- Deallocate the used memory
  for (i = 0; i < n; i++) {
    free(a[i]); free(b[i]); free(c[i]);
  }
  free(a); free(b); free(c);

  return;
}


//--
//-- Run a series of NxN Matrix Matrix multiplication
//--  using different stratagies of OpenMP threading
//--
void RunAllThreadTests (int n, int t) {

  //--
  //-- Variables used in this function
  //--
  double **a; double **b; double **c;
  int i; int j; int seed;
  double T; double R;

  //--
  //-- Allocate the storage for matrices.]
  //--
  a = (double **) calloc (n, sizeof (double));
  b = (double **) calloc (n, sizeof (double));
  c = (double **) calloc (n, sizeof (double));

  for (i = 0; i < n; i++) a[i] = (double *) calloc (n, sizeof (double));
  for (i = 0; i < n; i++) b[i] = (double *) calloc (n, sizeof (double));
  for (i = 0; i < n; i++) c[i] = (double *) calloc (n, sizeof (double));


  //--
  //-- Assign randomly generated values to the input matrices B and C.
  //--
  seed = 123456789;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      b[i][j] = randomize (&seed);
    }
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      c[i][j] = randomize (&seed);
    }
  }



  //######################################################
  //--
  //-- Run the Naive OpenMP-Static Test
  //--
  //######################################################
  naive_omp_s (n, t, a, b, c);

  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("naive_omp_s is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char naive_omp_s_desc[] = "Naive OpenMP Static.";
  stats(naive_omp_s_desc, n, t, &T, &R);

  avgTime_Naive_Static += T;
  avgRate_Naive_Static += R;


  //-- Clear out Mat A
  clear(n, a);





  //######################################################
  //--
  //-- Run the Naive OpenMP-Dynamic Test
  //--
  //######################################################
  naive_omp_d (n, t, a, b, c);

  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("naive_omp_d is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char naive_omp_d_desc[] = "Naive OpenMP Dynamic.";
  stats(naive_omp_d_desc, n, t, &T, &R);

  avgTime_Naive_Dynamic += T;
  avgRate_Naive_Dynamic += R;

  //-- Clear out Mat A
  clear(n, a);





  //######################################################
  //--
  //-- Run the Naive OpenMP-Guided Test
  //--
  //######################################################
  naive_omp_g (n, t, a, b, c);

  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("naive_omp_g is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char naive_omp_g_desc[] = "Naive OpenMP Guided.";
  stats(naive_omp_g_desc, n, t, &T, &R);

  avgTime_Naive_Guided += T;
  avgRate_Naive_Guided += R;

  //-- Clear out Mat A
  clear(n, a);





  //######################################################
  //--
  //-- Run the Optimized OpenMP Static Test
  //--
  //######################################################
  optim_omp_s(n, t, a, b, c);

  #if debug
  //-- Optional Validation
  if (validate (n, s ,a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("optim_omp_s is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char optim_omp_s_desc[] = "Loop Optimized OpenMP Static.";
  stats(optim_omp_s_desc, n, t, &T, &R);

  avgTime_Optim_Static += T;
  avgRate_Optim_Static += R;

  //-- Clear out Mat A
  clear(n, a);





  //######################################################
  //--
  //-- Run the Optimized OpenMP Dynamic Test
  //--
  //######################################################
  optim_omp_d(n, t, a, b, c);

  #if debug
  //-- Optional Validation
  if (validate (n, s ,a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("optim_omp_d is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char optim_omp_d_desc[] = "Loop Optimized OpenMP Dynamic.";
  stats(optim_omp_d_desc, n, t, &T, &R);

  avgTime_Optim_Dynamic += T;
  avgRate_Optim_Dynamic += R;

  //-- Clear out Mat A
  clear(n, a);





  //######################################################
  //--
  //-- Run the Optimized OpenMP Grouped Test
  //--
  //######################################################
  optim_omp_g(n, t, a, b, c);

  #if debug
  //-- Optional Validation
  if (validate (n, s ,a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("optim_omp_g is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char optim_omp_g_desc[] = "Loop Optimized OpenMP Guided.";
  stats(optim_omp_g_desc, n, t, &T, &R);

  avgTime_Optim_Guided += T;
  avgRate_Optim_Guided += R;

  //-- Clear out Mat A
  clear(n, a);





  //######################################################
  //--
  //-- Run a series of Serial Blocking Tests
  //--
  //######################################################
  block_omp_s(n, t, 16, a, b, c);

  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("Blocking Static is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char blocking_omp_s_desc[] = "Blocking OpenMP Static - 16.";
  stats(blocking_omp_s_desc, n, t, &T, &R);

  avgTime_Block_Static += T;
  avgRate_Block_Static += R;

  //-- Clear out Mat A
  clear(n, a);



  //######################################################
  //--
  //-- Run a series of Dynamic Blocking Tests
  //--
  //######################################################
  block_omp_d(n, t, 16, a, b, c);

  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("Blocking Dynamic is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char blocking_omp_d_desc[] = "Blocking OpenMP Dynamic - 16.";
  stats(blocking_omp_d_desc, n, t, &T, &R);

  avgTime_Block_Dynamic += T;
  avgRate_Block_Dynamic += R;

  //-- Clear out Mat A
  clear(n, a);






  //######################################################
  //--
  //-- Run a series of Group Blocking Tests
  //--
  //######################################################
  block_omp_g(n, t, 16, a, b, c);

  #if debug
  //-- Optional Validation
  if (validate (n, s, a)) {
    printf ("\n\n\n###################################\n\n\n");
    printf ("Blocking Group is incorrect!!");
    printf ("\n\n\n###################################\n\n\n");
    ErrorCount++;
  }
  #endif

  //-- Display Stats
  char blocking_omp_g_desc[] = "Blocking OpenMP Guided - 16.";
  stats(blocking_omp_g_desc, n, t, &T, &R);

  avgTime_Block_Guided += T;
  avgRate_Block_Guided += R;

  //-- Clear out Mat A
  clear(n, a);


  //-- Deallocate the used memory
  for (i = 0; i < n; i++) {
    free(a[i]); free(b[i]); free(c[i]);
  }
  free(a); free(b); free(c);

  return;
}


//--
//-- Get a randomized value, and refresh seed.
//--
double randomize (int *seed) {
  int k; double r;
  k = *seed / 127773;
  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;
  if ( *seed < 0 ) { *seed = *seed + 2147483647; }
  r = (double) (*seed) * 4.656612875E-10;
  return r;
}


//--
//-- clear out the contents of X
//--
void clear (int n, double **X) {
  int i ,j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      X[i][j] = 0.0;
    }
  }
}


//--
//-- compare the passed in matracies to see
//-- if there are any differences between them
//--
int validate (int n, double **S, double **X) {

  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (S[i][j] != X[i][j]) {
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
  double time, rate;

  ops  = (unsigned long long)n;
  ops *= (unsigned long long)n;
  ops *= (unsigned long long)n;
  ops *= 2;

  time = time_stop - time_begin;
  rate = ( double ) ( ops ) / time / 1000000.0;

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
  printf("Usage: ./MatMultOpenMP [-h] -n <num> -t <num> [-s] [-p]\n");
  printf("Options:\n");
  printf("  -h\t\tPrint this help message.\n");
  printf("  -n <num>\tSize of N.\n");
  printf("  -t <num>\tNumber of Threads.\n");
  printf("  -s\t\tRun all Serial Tests\n");
  printf("  -p\t\tRun all OpenMP Tests\n");
  printf("Examples:\n");
  printf("linux> ./MatMultOpenMP -n 1024 -t 8 -p\n");
}


//--
//--  Implementation of Different NxN Matrix Multiplication
//--

//--
//-- naive_serial : simple row by column for fixed A.
//--
//-- Notes : poor cache performance, serial
//--
void naive_serial (int n, double **A, double **B, double **C) {

  int i, j, k;

  time_begin = omp_get_wtime();

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        A[i][j] = A[i][j] + (B[i][k] * C[k][j]);
      }
    }
  }

  time_stop = omp_get_wtime();
}


//--
//-- naive_omp_s : row by column with OpenMP for fixed A.
//--
//-- notes : poor cache performance, using threading, static
//--
void naive_omp_s (int n, int t, double **A, double **B, double **C) {

  //--
  //-- Set the number of threads. This will be constant for all tests
  //--
  omp_set_num_threads(t);

  int i, j, k;

  time_begin = omp_get_wtime();

  #pragma omp parallel	  \
  shared (A, B, C, n)	\
  private (i, j, k)

  #pragma omp for schedule(static)
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        A[i][j] = A[i][j] + (B[i][k] * C[k][j]);
      }
    }
  }

  time_stop = omp_get_wtime();
}


//--
//-- naive_omp_d : row by column with OpenMP for fixed A.
//--
//-- notes : poor cache performance, using threading, dynamic
//--
void naive_omp_d (int n, int t, double **A, double **B, double **C) {

  //--
  //-- Set the number of threads. This will be constant for all tests
  //--
  omp_set_num_threads(t);

  int i, j, k;

  time_begin = omp_get_wtime();

  #pragma omp parallel	  \
  shared (A, B, C, n)	\
  private (i, j, k)

  #pragma omp for schedule(dynamic,DYNAMC_CHUNK)
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        A[i][j] = A[i][j] + (B[i][k] * C[k][j]);
      }
    }
  }

  time_stop = omp_get_wtime();
}


//--
//-- naive_omp_g : row by column with OpenMP for fixed A.
//--
//-- notes : poor cache performance, using threading, guided
//--
void naive_omp_g (int n, int t, double **A, double **B, double **C) {

  //--
  //-- Set the number of threads. This will be constant for all tests
  //--
  omp_set_num_threads(t);

  int i, j, k;

  time_begin = omp_get_wtime();

  #pragma omp parallel	  \
  shared (A, B, C, n)	\
  private (i, j, k)

  #pragma omp for schedule(guided)
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        A[i][j] = A[i][j] + (B[i][k] * C[k][j]);
      }
    }
  }

  time_stop = omp_get_wtime();
}


//--
//-- optim_serial : kij row by row with fixed B.
//--
//-- notes : good cache performance, serial
//--
void optim_serial (int n, double **A, double **B, double **C) {

  int i, j, k;
  double r;

  time_begin = omp_get_wtime();

  for (k = 0; k < n; k++) {
    for (i = 0; i < n; i++) {
      r = B[i][k];
      for (j = 0; j < n; j++) {
        A[i][j] += r * C[k][j];
      }
    }
  }

  time_stop = omp_get_wtime();
}


//--
//-- omptim_omp_s : kij row by row with OpenMP for fixed B
//--
//-- notes : good cache performance with threading, static
//--
void optim_omp_s (int n, int t, double **A, double **B, double **C) {

  //--
  //-- Set the number of threads. This will be constant for all tests
  //--
  omp_set_num_threads(t);

  int i, j, k;
  double r;

  time_begin = omp_get_wtime();

  #pragma omp parallel		  \
  shared (A, B, C, n)		\
  private (i, j, k, r)

  for (k = 0; k < n; k++) {
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
      r = B[i][k];
      for (j = 0; j < n; j++) {
        A[i][j] += r * C[k][j];
      }
    }
  }

  time_stop = omp_get_wtime();
}


//--
//-- omptim_omp_d : kij row by row with OpenMP for fixed B
//--
//-- notes : good cache performance with threading, dynamic
//--
void optim_omp_d (int n, int t, double **A, double **B, double **C) {

  //--
  //-- Set the number of threads. This will be constant for all tests
  //--
  omp_set_num_threads(t);

  int i, j, k;
  double r;

  time_begin = omp_get_wtime();

  # pragma omp parallel		  \
  shared (A, B, C, n)		\
  private (i, j, k, r)

  for (k = 0; k < n; k++) {
    # pragma omp for schedule(dynamic,DYNAMC_CHUNK)
    for (i = 0; i < n; i++) {
      r = B[i][k];
      for (j = 0; j < n; j++) {
        A[i][j] += r * C[k][j];
      }
    }
  }

  time_stop = omp_get_wtime();
}


//--
//-- omptim_omp_g : kij row by row with OpenMP for fixed B
//--
//-- notes : good cache performance with threading, guided
//--
void optim_omp_g (int n, int t, double **A, double **B, double **C) {

  //--
  //-- Set the number of threads. This will be constant for all tests
  //--
  omp_set_num_threads(t);

  int i, j, k;
  double r;

  time_begin = omp_get_wtime();

  # pragma omp parallel	  	\
  shared (A, B, C, n)		\
  private (i, j, k, r)

  for (k = 0; k < n; k++) {
    # pragma omp for schedule(guided)
    for (i = 0; i < n; i++) {
      r = B[i][k];
      for (j = 0; j < n; j++) {
        A[i][j] += r * C[k][j];
      }
    }
  }

  time_stop = omp_get_wtime();
}


//--
//-- block_serial : ijk based blocks
//--
//-- notes : compromise of temporal and spatial locality, serial
//--
void block_serial (int n, int b, double **A, double **B, double **C) {

  int i, j, k, en, jj, kk;
  double sum = 0.0;
  en = b * (n/b);

  time_begin = omp_get_wtime();

  for (kk = 0; kk < en; kk += b) {
    for (jj = 0; jj < en; jj += b) {
      for (i = 0; i < n; i++) {
        for (j = jj; j < jj+b; j++) {
          sum = A[i][j];
          for (k = kk; k < kk+b; k++) {
            sum += B[i][k] * C[k][j];
          }
          A[i][j] = sum;
        }
      }
    }
  }

  time_stop = omp_get_wtime();
}


//--
//-- block_omp_s : ijk based blocks with OpenMP
//--
//-- notes : compromise of temporal and spatial locality with threading, static
//--
void block_omp_s (int n, int t, int b, double **A, double **B, double **C) {

  //--
  //-- Set the number of threads. This will be constant for all tests
  //--
  omp_set_num_threads(t);

  int i, j, k, jj, kk;
  double sum = 0.0;
  int en = b * (n / b);

  time_begin = omp_get_wtime();

  #pragma omp parallel	      \
  shared  (A, B, C, n, en)		\
  private (i, j, k, jj, kk, sum)
  for (kk = 0; kk < en; kk += b) {
    for (jj = 0; jj < en; jj += b) {
      for (i = 0; i < n; i++) {
        #pragma omp for schedule(static)
        for (j = jj; j < jj+b; j++) {
          sum = A[i][j];
          for (k = kk; k < kk+b; k++) {
            sum += B[i][k] * C[k][j];
          }
          A[i][j] = sum;
        }
      }
    }
  }

  time_stop = omp_get_wtime();
}


//--
//-- block_omp_s : ijk based blocks with OpenMP
//--
//-- notes : compromise of temporal and spatial locality with threading, static
//--
void block_omp_d (int n, int t, int b, double **A, double **B, double **C) {

  //--
  //-- Set the number of threads. This will be constant for all tests
  //--
  omp_set_num_threads(t);

  int i, j, k, jj, kk;
  double sum = 0.0;
  int en = b * (n / b);

  time_begin = omp_get_wtime();

  #pragma omp parallel	      \
  shared  (A, B, C, n, en)		\
  private (i, j, k, jj, kk, sum)
  for (kk = 0; kk < en; kk += b) {
    for (jj = 0; jj < en; jj += b) {
      #pragma omp for schedule(dynamic,DYNAMC_CHUNK)
      for (i = 0; i < n; i++) {
        for (j = jj; j < jj+b; j++) {
          sum = A[i][j];
          for (k = kk; k < kk+b; k++) {
            sum += B[i][k] * C[k][j];
          }
          A[i][j] = sum;
        }
      }
    }
  }

  time_stop = omp_get_wtime();
}


//--
//-- block_omp_s : ijk based blocks with OpenMP
//--
//-- notes : compromise of temporal and spatial locality with threading, static
//--
void block_omp_g (int n, int t, int b, double **A, double **B, double **C) {

  //--
  //-- Set the number of threads. This will be constant for all tests
  //--
  omp_set_num_threads(t);

  int i, j, k, jj, kk;
  double sum = 0.0;
  int en = b * (n / b);

  time_begin = omp_get_wtime();

  #pragma omp parallel	      \
  shared  (A, B, C, n, en)		\
  private (i, j, k, jj, kk, sum)
  for (kk = 0; kk < en; kk += b) {
    for (jj = 0; jj < en; jj += b) {
      #pragma omp for schedule(guided)
      for (i = 0; i < n; i++) {
        for (j = jj; j < jj+b; j++) {
          sum = A[i][j];
          for (k = kk; k < kk+b; k++) {
            sum += B[i][k] * C[k][j];
          }
          A[i][j] = sum;
        }
      }
    }
  }

  time_stop = omp_get_wtime();
}
