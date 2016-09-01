#include "ASEArch_blas.h"
#include "safecall.h"
#include "cublas_v2.h"
#include <stdlib.h>
#include <stdio.h>

#define FWD_TOL 2e-14
#define BWD_TOL 1e-14

void NVIDIA_dtrsv(enum ASEArch_uplo uplo, enum ASEArch_trans trans,
   enum ASEArch_diag diag, int n, const double a[], int lda, double x[],
   int incx);

const char ftrans='T';
const char fnontrans='N';
const char flwr='L';
const char fupr='U';
const char fnonunit='N';
const char funit='U';

#ifdef HAVE_MAGMABLAS
extern "C"
void  magmablas_dtrsm( char side, char uplo, char tran, char diag,
      int M, int N, double alpha, /*const*/ double* A, int lda, double* b,
      int ldb);

void magma_dtrsv(const char *uplo, const char *trans, const char *diag,
      const int *n, /*const*/ double a[], const int *lda, double x[],
      const int *incx) {
   if(*incx!=1) {
      printf("can't use magma_dtrsv if incx!=1\n");
      return;
   }
   magmablas_dtrsm( 'L', *uplo, *trans, *diag, *n, 1, double(1.0), 
         a, *lda, x, *n);
}
#endif /* HAVE_MAGMABLAS */

#ifdef HAVE_CUBLAS
void dtrsv_cublas_wrapper(enum ASEArch_uplo uplo, enum ASEArch_trans trans, enum ASEArch_diag diag, int n, const double *a, int lda, double *x, int incx) {
   cublasHandle_t handle;
   cublasCreate(&handle);
   if(trans == ASEARCH_NONTRANS) {
      if(uplo == ASEARCH_LWR) {
         if(diag == ASEARCH_UNIT) {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                  CUBLAS_DIAG_UNIT, n, a, lda, x, incx);
         } else {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                  CUBLAS_DIAG_NON_UNIT, n, a, lda, x, incx);
         }
      } else {
         if(diag == ASEARCH_UNIT) {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                  CUBLAS_DIAG_UNIT, n, a, lda, x, incx);
         } else {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                  CUBLAS_DIAG_NON_UNIT, n, a, lda, x, incx);
         }
      }
   } else {
      if(uplo == ASEARCH_LWR) {
         if(diag == ASEARCH_UNIT) {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                  CUBLAS_DIAG_UNIT, n, a, lda, x, incx);
         } else {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                  CUBLAS_DIAG_NON_UNIT, n, a, lda, x, incx);
         }
      } else {
         if(diag == ASEARCH_UNIT) {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                  CUBLAS_DIAG_UNIT, n, a, lda, x, incx);
         } else {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                  CUBLAS_DIAG_NON_UNIT, n, a, lda, x, incx);
         }
      }
   }
   cublasDestroy(handle);
}
#endif /* HAVE_CUBLAS */

extern "C"
void dtrsv_(const char *uplo, const char *trans, const char *diag, const int *n,
      const double a[], const int *lda, double x[], const int *incx);
extern "C"
void dpotrf_(const char *uplo, const int *n, double a[], const int *lda,
      int *info);

void dtrsv_host_wrapper(enum ASEArch_uplo uplo, enum ASEArch_trans trans, enum ASEArch_diag diag, int n, const double *a, int lda, double *x, int incx) {
   if(trans == ASEARCH_NONTRANS) {
      if(uplo == ASEARCH_LWR) {
         if(diag == ASEARCH_UNIT) {
            dtrsv_(&flwr, &fnontrans, &funit, &n, a, &lda, x, &incx);
         } else {
            dtrsv_(&flwr, &fnontrans, &fnonunit, &n, a, &lda, x, &incx);
         }
      } else {
         if(diag == ASEARCH_UNIT) {
            dtrsv_(&fupr, &fnontrans, &funit, &n, a, &lda, x, &incx);
         } else {
            dtrsv_(&fupr, &fnontrans, &fnonunit, &n, a, &lda, x, &incx);
         }
      }
   } else {
      if(uplo == ASEARCH_LWR) {
         if(diag == ASEARCH_UNIT) {
            dtrsv_(&flwr, &ftrans, &funit, &n, a, &lda, x, &incx);
         } else {
            dtrsv_(&flwr, &ftrans, &fnonunit, &n, a, &lda, x, &incx);
         }
      } else {
         if(diag == ASEARCH_UNIT) {
            dtrsv_(&fupr, &ftrans, &funit, &n, a, &lda, x, &incx);
         } else {
            dtrsv_(&fupr, &ftrans, &fnonunit, &n, a, &lda, x, &incx);
         }
      }
   }
}

int test_variant(int n, const double alwr[], const double x[], int lda, int incx, enum ASEArch_trans trans, enum ASEArch_uplo uplo, enum ASEArch_diag diag);

int test_for_n(int n) {
   int errors=0;
   printf("Tests for n=%d\n", n);
   printf("==============\n");

   /* Generate a A and x ~ Unif(-1,1) */
   double *a = (double *) malloc(n*n*sizeof(double));
   for(int i=0; i<n*n; i++)
      a[i] = ((double) rand()) / RAND_MAX;
   double *x = (double *) malloc(n*sizeof(double));
   for(int i=0; i<n; i++)
      x[i] = ((double) rand()) / RAND_MAX;

   /* Note that random lower triangular matrices have exponetially bad
      conditions numbers. So perform Cholesky on A to get one that's better.
      Make diagonally dominant first so this is a valid maenouver. */
   for(int i=0; i<n ;i++) {
      for(int j=i+1; j<n; j++) {
         a[i*n+i] += a[i*n+j];
         a[j*n+j] += a[i*n+j];
      }
   }
   int info;
   dpotrf_(&flwr, &n, a, &n, &info);
   if(info!=0) {
      printf("DPOTRF Error info=%d\n", info);
      exit(1);
   }

   /* To pick up transitory errors, fill rest of matrix with NaNs */
   for(int i=0; i<n; i++)
      for(int j=0; j<i; j++)
         a[i*n+j] = nan("");

   /* Call for vareity of different parameters */
   for(int lda=n; lda<n+4; lda++) {
      for(int incx=-1; incx<4; incx++) {
         if(incx==0) continue;
         errors += test_variant(n, a, x, lda, incx, ASEARCH_NONTRANS, ASEARCH_LWR, ASEARCH_UNIT);
         errors += test_variant(n, a, x, lda, incx, ASEARCH_NONTRANS, ASEARCH_LWR, ASEARCH_NONUNIT);
         errors += test_variant(n, a, x, lda, incx, ASEARCH_TRANS, ASEARCH_LWR, ASEARCH_UNIT);
         errors += test_variant(n, a, x, lda, incx, ASEARCH_TRANS, ASEARCH_LWR, ASEARCH_NONUNIT);
         errors += test_variant(n, a, x, lda, incx, ASEARCH_NONTRANS, ASEARCH_UPR, ASEARCH_UNIT);
         errors += test_variant(n, a, x, lda, incx, ASEARCH_NONTRANS, ASEARCH_UPR, ASEARCH_NONUNIT);
         errors += test_variant(n, a, x, lda, incx, ASEARCH_TRANS, ASEARCH_UPR, ASEARCH_UNIT);
         errors += test_variant(n, a, x, lda, incx, ASEARCH_TRANS, ASEARCH_UPR, ASEARCH_NONUNIT);
      }
   }
   printf("\n\n");

   return errors;
}

void dgemv_for_test(enum ASEArch_uplo uplo, enum ASEArch_trans trans, enum ASEArch_diag diag, int n, const double *a, int lda, const double *x, int incx, double *b, int incb);

/* Run test on specified variant. Always expects A to be lwr on input with
   leading dimension lda. */
int test_variant(int n, const double alwr[], const double x[], int lda, int incx, enum ASEArch_trans trans, enum ASEArch_uplo uplo, enum ASEArch_diag diag) {
   int errors = 0;

   printf("* ");
   if(uplo==ASEARCH_UPR) printf("U");
   else printf("L");
   if(trans==ASEARCH_TRANS) printf("^T");
   printf("x=b, ");
   if(diag==ASEARCH_NONUNIT) printf("non-");
   printf("unit diag, lda=n");
   if(lda!=n) printf("+%d", lda-n);
   printf(", incx=%d... ", incx);

   /* Convert a to upper triangular/different lda if necessary */
   double *a = (double *) alwr;
   if(uplo == ASEARCH_UPR) {
      a = (double *) malloc(n*lda*sizeof(double));
      for(int i=0; i<n; i++)
         for(int j=0; j<n; j++)
            a[i*lda+j] = alwr[j*n+i];
   } else if(lda!=n) {
      a = (double *) malloc(n*lda*sizeof(double));
      for(int i=0; i<n; i++)
         for(int j=0; j<n; j++)
            a[i*lda+j] = alwr[i*n+j];
   }

   /* Calculate right hand side b = A*x */
   double *b = (double *) malloc(n*abs(incx)*sizeof(double));
   dgemv_for_test(uplo, trans, diag, n, a, lda, x, (incx<0)?-1:1, b, incx);

   /* Transfer data to device */
   double *a_gpu, *x_gpu;
   CudaSafeCall( cudaMalloc(&a_gpu, n*lda*sizeof(double)) );
   CudaSafeCall(
         cudaMemcpy(a_gpu, a, n*lda*sizeof(double), cudaMemcpyHostToDevice)
         );
   CudaSafeCall( cudaMalloc(&x_gpu, n*abs(incx)*sizeof(double)) );
   CudaSafeCall(
         cudaMemcpy(x_gpu, b, n*abs(incx)*sizeof(double), cudaMemcpyHostToDevice)
         );


   /* Perform test */
   /*double *soln = (double *) malloc(n*abs(incx)*sizeof(double));
   for(int i=0; i<n; i++) soln[i*abs(incx)] = b[i*abs(incx)];*/
   //dtrsv_host_wrapper(uplo, trans, diag, n, a, lda, soln, incx);
   ASEArch_dtrsv(uplo, trans, diag, n, a_gpu, lda, x_gpu, incx);
   //NVIDIA_dtrsv(uplo, trans, diag, n, a_gpu, lda, x_gpu, incx);

   /* Copy data back from GPU */
   double *soln = (double *) malloc(n*abs(incx)*sizeof(double));
   CudaSafeCall(
         cudaMemcpy(soln, x_gpu, n*abs(incx)*sizeof(double), cudaMemcpyDeviceToHost)
         );

   /* Test forward error */
   double fwd = 0;
   for(int i=0; i<n; i++)
      fwd += abs(soln[i*abs(incx)] - x[i])/n;
   if(fwd < FWD_TOL) {
      printf("fwd ok... ");
   } else {
      printf("FWD=%10.2e... ", fwd);
      /*for(int i=0; i<n; i++)
         printf("%d: %12.2e %12.2e diff %12.2e\n", i, soln[i*abs(incx)], x[i],
               abs(x[i]-soln[i*abs(incx)]));*/
      errors++;
   }

   /* Test backward error */
   double *resid = (double *) malloc(n*sizeof(double));
   dgemv_for_test(uplo, trans, diag, n, a, lda, soln, incx, resid, (incx<0)?-1:1);
   for(int i=0; i<n; i++) resid[i] -= b[i*abs(incx)];
   double rinf = 0;
   for(int i=0; i<n; i++) rinf = (rinf > resid[i]) ? rinf : resid[i];
   free(resid);
   double xinf = 0;
   for(int i=0; i<n; i++) xinf = (xinf > soln[i*abs(incx)]) ? xinf : soln[i*abs(incx)];
   double binf = 0;
   for(int i=0; i<n; i++) binf = (binf > b[i*abs(incx)]) ? binf : b[i*abs(incx)];
   double ainf = 0; /* Note: Using inf norm of alwr non-unit rather than a */
   for(int i=0; i<n; i++) {
      double rowsum=0;
      for(int j=0; j<=i; j++) {
         rowsum += alwr[j*n+i];
      }
      ainf = (ainf > rowsum) ? ainf : rowsum;
   }
   double bwd = rinf / (ainf * xinf + binf);
   if(bwd < BWD_TOL) {
      printf("bwd ok.\n");
   } else {
      printf("BWD=%10.2e.\n", bwd);
      errors++;
   }

   /* Cleanup memory */
   CudaSafeCall( cudaFree(a_gpu) );
   CudaSafeCall( cudaFree(x_gpu) );
   if(a!=alwr) free(a);
   free(b); free(soln);

   if(errors) exit(1);

   return errors;
}

void dgemv_for_test(enum ASEArch_uplo uplo, enum ASEArch_trans trans, enum ASEArch_diag diag, int n, const double *a, int lda, const double *x, int incx, double *b, int incb) {

   if(incx<0) x += (n-1)*(-incx); // start at last element if incx negative
   if(incb<0) b += (n-1)*(-incb); // start at last element if incb negative

   for(int i=0; i<n; i++) b[i*incb] = 0;
   if(trans == ASEARCH_NONTRANS) {
      if(uplo == ASEARCH_LWR) {
         if(diag == ASEARCH_NONUNIT) {
            for(int i=0; i<n; i++)
               for(int j=i; j<n; j++)
                  b[j*incb] += a[i*lda+j] * x[i*incx];
         } else { /* diag == ASEARCH_UNIT */
            for(int i=0; i<n; i++) {
               b[i*incb] += x[i*incx];
               for(int j=i+1; j<n; j++)
                  b[j*incb] += a[i*lda+j] * x[i*incx];
            }
         }
      } else {
         if(diag == ASEARCH_NONUNIT) {
            for(int i=0; i<n; i++)
               for(int j=0; j<=i; j++)
                  b[j*incb] += a[i*lda+j] * x[i*incx];
         } else { /* diag == ASEARCH_UNIT */
            for(int i=0; i<n; i++) {
               b[i*incb] += x[i*incx];
               for(int j=0; j<i; j++)
                  b[j*incb] += a[i*lda+j] * x[i*incx];
            }
         }
      }
   } else { /* trans == ASEARCH_TRANS */
      if(uplo == ASEARCH_LWR) {
         if(diag == ASEARCH_NONUNIT) {
            for(int i=0; i<n; i++)
               for(int j=i; j<n; j++)
                  b[i*incb] += a[i*lda+j] * x[j*incx];
         } else { /* diag == ASEARCH_UNIT */
            for(int i=0; i<n; i++) {
               b[i*incb] += x[i*incx];
               for(int j=i+1; j<n; j++)
                  b[i*incb] += a[i*lda+j] * x[j*incx];
            }
         }
      } else {
         if(diag == ASEARCH_NONUNIT) {
            for(int i=0; i<n; i++)
               for(int j=0; j<=i; j++)
                  b[i*incb] += a[i*lda+j] * x[j*incx];
         } else { /* diag == ASEARCH_UNIT */
            for(int i=0; i<n; i++) {
               b[i*incb] += x[i*incx];
               for(int j=0; j<i; j++)
                  b[i*incb] += a[i*lda+j] * x[j*incx];
            }
         }
      }
   }
}

int main(int argc, char *argv[]) {
   int errors=0;

   errors += test_for_n(10);
   errors += test_for_n(90);
   errors += test_for_n(200);
   errors += test_for_n(1000);
   errors += test_for_n(4001);

   if(errors==0) {
      printf("All tests succeeded.\n");
   } else {
      printf("Encountered %d errors\n", errors);
   }

   return errors;
}
