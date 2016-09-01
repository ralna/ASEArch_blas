#define USE_MAGMA
#define USE_CULA


#include "ASEArch_blas.h"
#include "safecall.h"
#include "cublas_v2.h"
#ifdef USE_CULA
#include "cula_blas_device.h"
#endif
#include <stdlib.h>
#include <stdio.h>

const char ftrans='T';
const char fnontrans='N';
const char flwr='L';
const char fupr='U';
const char fnonunit='N';
const char funit='U';

#ifdef USE_MAGMA
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
#endif

#ifdef USE_CULA
void cula_dtrsv(const char *uplo, const char *trans, const char *diag,
      const int *n, /*const*/ double a[], const int *lda, double x[],
      const int *incx) {
   if(*incx!=1) {
      printf("can't use cula_dtrsv if incx!=1\n");
      return;
   }
   int status = culaDeviceDtrsm( 'L', *uplo, *trans, *diag, *n, 1, double(1.0), 
         a, *lda, x, *n);
   if(status!=0) printf("culaStatus = %d\n", status);
}
#endif


extern "C"
void dtrsv_(const char *uplo, const char *trans, const char *diag, const int *n,
      const double a[], const int *lda, double x[], const int *incx);
extern "C"
void dpotrf_(const char *uplo, const int *n, double a[], const int *lda,
      int *info);
extern "C"
void dcopy_(const int *n, double *x, int *incx, double *y, int *incy);

double chkbwd(int n, const double a[], int lda, double x[], double b[]);

float inline tdiff(struct timespec tp1, struct timespec tp2) {
   return tp2.tv_sec - tp1.tv_sec +
      1e-9 * (tp2.tv_nsec-tp1.tv_nsec);
}

void __global__ warmUp(double x[]) {
   x[threadIdx.x] *=2;
}

void proc_arg(int argc, char **argv, int &n, ASEArch_uplo &uplo, 
      ASEArch_trans &trans, ASEArch_diag &diag);

int main(int argc, char *argv[]) {

   struct timespec tp1, tp2;

   int incx = 1;
   enum ASEArch_trans trans;
   enum ASEArch_uplo uplo;
   enum ASEArch_diag diag;

   /* Determine matrix size */
   int n;
   proc_arg(argc, argv, n, uplo, trans, diag);
   if(n==0) exit(0);
   printf("Running with a %d x %d matrix.\n", n, n);
   if(uplo==ASEARCH_UPR) printf("U");
   else printf("L");
   if(trans==ASEARCH_TRANS) printf("^T");
   printf("x=b, ");
   if(diag==ASEARCH_NONUNIT) printf("non-");
   printf("unit diag, lda=n");
   printf(", incx=%d.\n ", 1);

   /* Generate a A and x ~ Unif(-1,1) */
   double *a = (double *) malloc(n*n*sizeof(double));
   for(int i=0; i<n*n; i++)
      a[i] = ((double) rand()) / RAND_MAX;
   double *xorig = (double *) malloc(n*sizeof(double));
   for(int i=0; i<n; i++)
      xorig[i] = ((double) rand()) / RAND_MAX;
   double *x = (double *) malloc(n*sizeof(double));
   for(int i=0; i<n; i++)
      x[i] = xorig[i];

   /* Note that random lower triangular matrices have exponetially bad
      conditions numbers. So perform Cholesky on A to get one that's better.
      Make diagonally dominant so this is a valid maenouver. */
   for(int i=0; i<n ;i++) {
      for(int j=i+1; j<n; j++) {
         a[i*n+i] += a[i*n+j];
         a[j*n+j] += a[i*n+j];
      }
   }
   int info;
   if(uplo==ASEARCH_LWR) dpotrf_(&flwr, &n, a, &n, &info);
   else                  dpotrf_(&fupr, &n, a, &n, &info);
   if(info!=0) {
      printf("DPOTRF Error info=%d\n", info);
      exit(1);
   }

   /* Fill unused part of array with NaNs */
   if(uplo==ASEARCH_LWR) {
      for(int i=0; i<n; i++)
         for (int j=0; j<i; j++)
            a[i*n+j] = nan("");
   } else {
      for(int i=0; i<n; i++)
         for (int j=i+1; j<n; j++)
            a[i*n+j] = nan("");
   }

   /* Transfer data to device */
   clock_gettime(CLOCK_REALTIME, &tp1);
   double *a_gpu, *x_gpu, *xwork_gpu;
   CudaSafeCall( cudaMalloc(&a_gpu, n*n*sizeof(double)) );
   CudaSafeCall(
         cudaMemcpy(a_gpu, a, n*n*sizeof(double), cudaMemcpyHostToDevice)
         );
   CudaSafeCall( cudaMalloc(&x_gpu, n*sizeof(double)) );
   CudaSafeCall(
         cudaMemcpy(x_gpu, xorig, n*sizeof(double), cudaMemcpyHostToDevice)
         );
   clock_gettime(CLOCK_REALTIME, &tp2);
   printf("Transfer to device took %f\n", tdiff(tp1, tp2));
   CudaSafeCall( cudaMalloc(&xwork_gpu, n*sizeof(double)) );

   /* Warm up device */
   double *dummy;
   CudaSafeCall( cudaMalloc(&dummy, sizeof(double)) );
   warmUp <<<1,1>>>(dummy);

   int lda = n;

   /* Call CPU BLAS */
   clock_gettime(CLOCK_REALTIME, &tp1);
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
   clock_gettime(CLOCK_REALTIME, &tp2);
   printf("CPU BLAS took    %f\n", tdiff(tp1, tp2));

   /* Call CUBLAS BLAS */
   cublasHandle_t handle;
   cublasCreate(&handle);
   CudaSafeCall(
         cudaMemcpy(xwork_gpu, x_gpu, n*sizeof(double),
            cudaMemcpyDeviceToDevice)
         );
   cudaThreadSynchronize();
   clock_gettime(CLOCK_REALTIME, &tp1);
   if(trans == ASEARCH_NONTRANS) {
      if(uplo == ASEARCH_LWR) {
         if(diag == ASEARCH_UNIT) {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                  CUBLAS_DIAG_UNIT, n, a_gpu, lda, xwork_gpu, incx);
         } else {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                  CUBLAS_DIAG_NON_UNIT, n, a_gpu, lda, xwork_gpu, incx);
         }
      } else {
         if(diag == ASEARCH_UNIT) {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                  CUBLAS_DIAG_UNIT, n, a_gpu, lda, xwork_gpu, incx);
         } else {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                  CUBLAS_DIAG_NON_UNIT, n, a_gpu, lda, xwork_gpu, incx);
         }
      }
   } else {
      if(uplo == ASEARCH_LWR) {
         if(diag == ASEARCH_UNIT) {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                  CUBLAS_DIAG_UNIT, n, a_gpu, lda, xwork_gpu, incx);
         } else {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                  CUBLAS_DIAG_NON_UNIT, n, a_gpu, lda, xwork_gpu, incx);
         }
      } else {
         if(diag == ASEARCH_UNIT) {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                  CUBLAS_DIAG_UNIT, n, a_gpu, lda, xwork_gpu, incx);
         } else {
            cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                  CUBLAS_DIAG_NON_UNIT, n, a_gpu, lda, xwork_gpu, incx);
         }
      }
   }
   cudaThreadSynchronize();
   clock_gettime(CLOCK_REALTIME, &tp2);
   cublasDestroy(handle);
   printf("CUBLAS BLAS took %f\n", tdiff(tp1, tp2));

#ifdef USE_CULA
   /* Call CULA Dense BLAS */
   CudaSafeCall(
         cudaMemcpy(xwork_gpu, x_gpu, n*sizeof(double),
            cudaMemcpyDeviceToDevice)
         );
   culaInitialize();
   cudaThreadSynchronize();
   clock_gettime(CLOCK_REALTIME, &tp1);
   if(trans == ASEARCH_NONTRANS) {
      if(uplo == ASEARCH_LWR) {
         if(diag == ASEARCH_UNIT) {
            cula_dtrsv(&flwr, &fnontrans, &funit, &n, a_gpu, &lda, 
                  xwork_gpu, &incx);
         } else {
            cula_dtrsv(&flwr, &fnontrans, &fnonunit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         }
      } else {
         if(diag == ASEARCH_UNIT) {
            cula_dtrsv(&fupr, &fnontrans, &funit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         } else {
            cula_dtrsv(&fupr, &fnontrans, &fnonunit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         }
      }
   } else {
      if(uplo == ASEARCH_LWR) {
         if(diag == ASEARCH_UNIT) {
            cula_dtrsv(&flwr, &ftrans, &funit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         } else {
            cula_dtrsv(&flwr, &ftrans, &fnonunit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         }
      } else {
         if(diag == ASEARCH_UNIT) {
            cula_dtrsv(&fupr, &ftrans, &funit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         } else {
            cula_dtrsv(&fupr, &ftrans, &fnonunit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         }
      }
   }
   cudaThreadSynchronize();
   clock_gettime(CLOCK_REALTIME, &tp2);
   culaShutdown();
   printf("CULA Dense BLAS took %f\n", tdiff(tp1, tp2));
#endif


   /* Call MAGMA BLAS */
#ifdef USE_MAGMA
   CudaSafeCall(
         cudaMemcpy(xwork_gpu, x_gpu, n*sizeof(double),
            cudaMemcpyDeviceToDevice)
         );
   cudaThreadSynchronize();
   clock_gettime(CLOCK_REALTIME, &tp1);
   if(trans == ASEARCH_NONTRANS) {
      if(uplo == ASEARCH_LWR) {
         if(diag == ASEARCH_UNIT) {
            magma_dtrsv(&flwr, &fnontrans, &funit, &n, a_gpu, &lda, 
                  xwork_gpu, &incx);
         } else {
            magma_dtrsv(&flwr, &fnontrans, &fnonunit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         }
      } else {
         if(diag == ASEARCH_UNIT) {
            magma_dtrsv(&fupr, &fnontrans, &funit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         } else {
            magma_dtrsv(&fupr, &fnontrans, &fnonunit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         }
      }
   } else {
      if(uplo == ASEARCH_LWR) {
         if(diag == ASEARCH_UNIT) {
            magma_dtrsv(&flwr, &ftrans, &funit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         } else {
            magma_dtrsv(&flwr, &ftrans, &fnonunit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         }
      } else {
         if(diag == ASEARCH_UNIT) {
            magma_dtrsv(&fupr, &ftrans, &funit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         } else {
            magma_dtrsv(&fupr, &ftrans, &fnonunit, &n, a_gpu, &lda,
                  xwork_gpu, &incx);
         }
      }
   }
   cudaThreadSynchronize();
   clock_gettime(CLOCK_REALTIME, &tp2);
   printf("MAGMA BLAS took %f\n", tdiff(tp1, tp2));
#endif

   /* Call Device BLAS */
   CudaSafeCall(
         cudaMemcpy(xwork_gpu, x_gpu, n*sizeof(double),
            cudaMemcpyDeviceToDevice)
         );
   cudaThreadSynchronize();
   clock_gettime(CLOCK_REALTIME, &tp1);
   ASEArch_dtrsv(uplo, trans, diag, n, a_gpu, lda, xwork_gpu, incx);
   cudaThreadSynchronize();
   clock_gettime(CLOCK_REALTIME, &tp2);
   printf("Device BLAS took %f\n", tdiff(tp1, tp2));
   printf("(%f GB/s)\n", ((n*(n+1)/2)+n)*8.0/(1000*1000*1000*(tdiff(tp1, tp2))));

   /* Compare result */
   clock_gettime(CLOCK_REALTIME, &tp1);
   double *x2 = (double *) malloc(n*sizeof(double));
   CudaSafeCall(
         cudaMemcpy(x2, xwork_gpu, n*sizeof(double), cudaMemcpyDeviceToHost)
         );
   clock_gettime(CLOCK_REALTIME, &tp2);
   CudaCheckError();
   printf("Transfer from device took %f\n", tdiff(tp1, tp2));
   double diff=0;
   for(int i=0; i<n; i++) {
      diff += abs(x[i]-x2[i]);
      //if(!(abs(x[i]-x2[i])<1e-15)) printf("x[%d] = %10.2e %10.2e diff %10.2e\n", i, x[i], x2[i], abs(x[i]-x2[i]));
   }
   printf("Diff vs CPU = %le (%le abs)\n", diff/n, diff);

   /* Find dcopy speed on half matrix GPU */
   cublasCreate(&handle);
   cudaThreadSynchronize();
   clock_gettime(CLOCK_REALTIME, &tp1);
   cublasDcopy(handle, n*n/2, a_gpu, 1, a_gpu + n*n/2, 1);
   //CudaSafeCall(
   //      cudaMemcpy(a_gpu+n*n/2, a, sizeof(double)*n*n/2,
   //         cudaMemcpyDeviceToDevice)
   //      );
   cudaThreadSynchronize();
   clock_gettime(CLOCK_REALTIME, &tp2);
   cublasDestroy(handle);
   printf("nvidia dcopy took %f\n", tdiff(tp1, tp2));
   printf("C2070 DCOPY Speed %f\n", (2e-9*sizeof(double)*n*n/2)/tdiff(tp1, tp2));

   /* Find dcopy speed on half matrix Host */
   clock_gettime(CLOCK_REALTIME, &tp1);
   int na = n*n/2;
   int i1 = 1;
   dcopy_(&na, a, &i1, a + na, &i1);
   //memcpy(a+na, a, sizeof(double)*na);
   clock_gettime(CLOCK_REALTIME, &tp2);
   printf("host dcopy took %f\n", tdiff(tp1, tp2));
   printf("Host DCOPY Speed %f\n", (2e-9*sizeof(double)*n*n/2)/tdiff(tp1, tp2));


   /* Cleanup */
   CudaSafeCall( cudaFree(a_gpu) );
   CudaSafeCall( cudaFree(x_gpu) );
   CudaSafeCall( cudaFree(xwork_gpu) );
   free(a); free(x); free(x2);

}

void proc_arg(int argc, char **argv, int &n, ASEArch_uplo &uplo, 
      ASEArch_trans &trans, ASEArch_diag &diag) {

   /* Defaults */
   trans = ASEARCH_NONTRANS;
   uplo = ASEARCH_LWR;
   diag = ASEARCH_UNIT;
   n = 1000;

   /* Process args */
   char *prog_name = *argv;
   argv++;
   for(int i=1; i<argc; i++) {
      //printf("Processing %d = '%s'\n", i, *argv);
      if(0==strcmp(*argv, "-h") || 0==strcmp(*argv, "--help")) {
         printf("Usage:\n\t%s [-h] [--trans|--non-trans] [--unit|--non-unit] [--upper|--lower] [n]\n\n", prog_name);
         n = 0;
      }
      else if(0==strcmp(*argv, "--trans")) {
         trans = ASEARCH_TRANS;
      }
      else if(0==strcmp(*argv, "--non-trans")) {
         trans = ASEARCH_NONTRANS;
      }
      else if(0==strcmp(*argv, "--unit")) {
         diag = ASEARCH_UNIT;
      }
      else if(0==strcmp(*argv, "--non-unit")) {
         diag = ASEARCH_NONUNIT;
      }
      else if(0==strcmp(*argv, "--upper")) {
         uplo = ASEARCH_UPR;
      }
      else if(0==strcmp(*argv, "--lower")) {
         uplo = ASEARCH_LWR;
      }
      else {
         n = atoi(*argv);
      }
      argv++;
   }
}
