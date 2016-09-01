#include "ASEArch_blas.h"
#include <stdio.h>

/* Following kernel *must* be launched with exactly one row per an x thread */
template <typename T_ELEM, unsigned int threadsx, unsigned int threadsy>
void __global__ ASEArch_gemv_notrans(int m, int n, T_ELEM alpha,
      const T_ELEM *a, int lda, const T_ELEM *x, int incx, T_ELEM beta,
      T_ELEM *y, int incy) {

   volatile T_ELEM __shared__ cache[threadsy*threadsx];

   if(incx<0) x += (n-1)*(-incx);
   if(incy<0) y += (m-1)*(-incy);

   if(threadIdx.x+blockIdx.x*threadsx>=m) return;
   y += int(blockIdx.x*threadsx+threadIdx.x)*incy;
   a += blockIdx.x*threadsx+threadIdx.x;
   T_ELEM val = 0;
   for(int j=threadIdx.y; j<n; j+=threadsy) { // col
      val += a[j*lda] * x[j*incx];
   }
   cache[threadIdx.x*threadsy+threadIdx.y] = alpha*val;
   __syncthreads();
   if(threadIdx.y==0) {
      y[0] = beta*y[0] + alpha*val;
      for(int i=1; i<threadsy; i++)
         y[0] += cache[threadIdx.x*threadsy+i];
   }
}

template <typename T_ELEM, unsigned int width>
__device__ T_ELEM warpReduce(unsigned int tid, volatile T_ELEM *cache) {
   if(width>64) printf("warpReduce with width >64!");
   if(tid >= width/2) return 0.0;
   if(width>=64) cache[tid] += cache[tid+32];
   if(width>=32) cache[tid] += cache[tid+16];
   if(width>=16) cache[tid] += cache[tid+8];
   if(width>=8) cache[tid] += cache[tid+4];
   if(width>=4) cache[tid] += cache[tid+2];
   if(width>=2) {
      return cache[tid] + cache[tid+1];
   } else {
      return cache[tid];
   }
}

#define DGEMVT_THREADSX 16 // was 8 but 16 better for 125x1000
#define DGEMVT_THREADSY 4 // was 16 but 4 better for 125x1000
#define ROWS_PER_THREAD 1
#define COLS_PER_THREAD 1
#define ROWS_PER_BLOCK (ROWS_PER_THREAD*DGEMVT_THREADSY)

template <typename T_ELEM, unsigned int threadsx, unsigned int threadsy, unsigned int cpt>
void __global__ ASEArch_gemv_trans(int m, int n, T_ELEM alpha,
      const T_ELEM *a, int lda, const T_ELEM *x, int incx, T_ELEM beta,
      T_ELEM *y, int incy) {

   volatile T_ELEM __shared__ cache[threadsx*threadsy];

   if(incx<0) x += (m-1)*(-incx);
   if(incy<0) y += (n-1)*(-incy);

   int i = blockIdx.y*ROWS_PER_BLOCK + threadIdx.y;
   if(i>=n) return;
   T_ELEM val = 0, val2;
   if(cpt>=2) val2 = 0;
   for(int j=threadIdx.x; j<m; j+=cpt*threadsx) {
      val += a[i*lda+j] * x[j*incx];
      if(cpt>=2 && j+threadsx<m)
         val2 += a[i*lda+j+threadsx] * x[int(j+threadsx)*incx];
   }
   switch(cpt) {
   case(1):
      cache[threadIdx.y*threadsx+threadIdx.x] = val;
      break;
   case(2):
      cache[threadIdx.y*threadsx+threadIdx.x] = val + val2;
      val += val2;
      break;
   }
   //val = alpha*warpReduce<threadsx>(threadIdx.x, &cache[threadIdx.y*threadsx]);
   if(threadIdx.x==0) {
      for(int k=1; k<threadsx; k++)
         val += cache[threadIdx.y*threadsx+k];
      y[i*incy] = beta*y[i*incy] + alpha*val;
   }
}

#define DGEMVN_THREADSX 16
#define DGEMVN_THREADSY 8

/*
   USE ALL DEVICE POINTERS!
   Performs the matrix vector product:
   y = alpha * A * x + beta * y,   or   y = alpha * A^T * x + beta * y
*/
void ASEArch_dgemv(enum ASEArch_trans trans, int m, int n, double alpha,
      const double a[], int lda, const double x[], int incx, double beta,
      double y[], int incy) {

   if(m<=0 || n<=0) return;

   if(trans == ASEARCH_TRANS) {
      dim3 nblocks(1,1);
      nblocks.y = (n-1)/ROWS_PER_BLOCK + 1;
      dim3 nthreads(DGEMVT_THREADSX, DGEMVT_THREADSY);
      size_t shmem = 0;
      ASEArch_gemv_trans <double, DGEMVT_THREADSX, DGEMVT_THREADSY, COLS_PER_THREAD>
         <<< nblocks, nthreads, shmem >>>
         (m, n, alpha, a, lda, x, incx, beta, y, incy);
   } else {
      cudaFuncSetCacheConfig(
            ASEArch_gemv_notrans<double, DGEMVN_THREADSX,DGEMVN_THREADSY>,
            cudaFuncCachePreferL1);
      dim3 nblocks(1,1);
      nblocks.x = (m-1) / DGEMVN_THREADSX + 1;
      dim3 nthreads(DGEMVN_THREADSX, DGEMVN_THREADSY);
      size_t shmem = 0;
      ASEArch_gemv_notrans <double, DGEMVN_THREADSX, DGEMVN_THREADSY>
         <<< nblocks, nthreads, shmem >>>
         (m, n, alpha, a, lda, x, incx, beta, y, incy);
   }
}

/*
   USE ALL DEVICE POINTERS!
   Performs the matrix vector product:
   y = alpha * A * x + beta * y,   or   y = alpha * A^T * x + beta * y
*/
void ASEArch_sgemv(enum ASEArch_trans trans, int m, int n, float alpha,
      const float a[], int lda, const float x[], int incx, float beta,
      float y[], int incy) {

   if(m<=0 || n<=0) return;

   if(trans == ASEARCH_TRANS) {
      dim3 nblocks(1,1);
      nblocks.y = (n-1)/ROWS_PER_BLOCK + 1;
      dim3 nthreads(DGEMVT_THREADSX, DGEMVT_THREADSY);
      size_t shmem = 0;
      ASEArch_gemv_trans <float, DGEMVT_THREADSX, DGEMVT_THREADSY, COLS_PER_THREAD>
         <<< nblocks, nthreads, shmem >>>
         (m, n, alpha, a, lda, x, incx, beta, y, incy);
   } else {
      cudaFuncSetCacheConfig(
            ASEArch_gemv_notrans<float, DGEMVN_THREADSX,DGEMVN_THREADSY>,
            cudaFuncCachePreferL1);
      dim3 nblocks(1,1);
      nblocks.x = (m-1) / DGEMVN_THREADSX + 1;
      dim3 nthreads(DGEMVN_THREADSX, DGEMVN_THREADSY);
      size_t shmem = 0;
      ASEArch_gemv_notrans <float, DGEMVN_THREADSX, DGEMVN_THREADSY>
         <<< nblocks, nthreads, shmem >>>
         (m, n, alpha, a, lda, x, incx, beta, y, incy);
   }
}
