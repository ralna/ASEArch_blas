/*
Copyright (c) 2012-13, The Science and Technology Facilities Council (STFC)
Copyright (c) 2012, NVIDIA
Principal Author: Jonathan Hogg (STFC)
Other Contributors: 
   Christopher Munro (STFC)
   Philippe Vandermersch (NVIDIA)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the STFC nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE STFC BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

__inline__ __device__ float cuConj(float x) { return x; }
__inline__ __device__ double cuConj(double x) { return x; }

template<typename T_ELEM>
__inline__ __device__ T_ELEM loadVolatile(volatile T_ELEM *vptr) {
   return *vptr;
}

#include "ASEArch_blas.h"
#include "safecall.h"
#include <stdio.h>
#include <cuComplex.h>

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

/* If INV_AFTER is defined, then for the global memory variant, an explicit
   backwards stable inverse is calculated for the diagonal blocks of row
   INV_AFTER and all subsequent rows */
#define INV_AFTER 4

/* 
 * Global memory variant parameters:
 * NB_TASK is rows per thread block = numThreads.x
 * THREADSY_TASK is numThreads.y
 */
#define NB_TASK 32 // Strongly recommend = warpSize
#define THREADSX_TASK NB_TASK // This MUST == NB_TASK <= warpSize
#ifndef THREADSY_TASK
#define THREADSY_TASK 4
#endif

/*
 * Kernel-synchronized variant parameters
 * Thread block uses THREADSX x THREADSY threads
 * NBI is inner block size, shouldn't be larger than warpSize or bad things
 *    will happen
 * NB is outer block size, should be multiple of inner block size
 */
#ifndef THREADSX
#define THREADSX 96
#endif
#ifndef THREADSY
#define THREADSY 4
#endif
#ifndef NBI
#define NBI 32       // Inner block size: max warpSize
#endif
#ifndef NB
#define NB (4*NBI)   // Outer block size: used for thread blocks
#endif

/* Use global memory variant if n >= SWITCH_TO_GMEM */
#define SWITCH_TO_GMEM 256

/* Uncomment following line to enable profiling */
//#define TIMING

unsigned int __inline__ __device__ getSM(void) {
   unsigned int output;
   asm("mov.u32 %0,%smid;" : "=r"(output) : );
   return output;
}

/* Performs trsv on a blksz x blksz tile.
 * IMPORTANT blkSize <= warpSize
 */
template <typename T_ELEM, int blkSize, bool ISUNIT>
void __device__ dblkSolve(const T_ELEM *minus_a, int lda, T_ELEM &val) {

   volatile T_ELEM __shared__ xs;

#pragma unroll 16
   for(int i=0; i<blkSize; i++) {
      if(threadIdx.x==i) {
         if(!ISUNIT) val *= minus_a[i*lda+i];
         xs = val;
      }
      if(threadIdx.x>i)
         val += minus_a[i*lda+threadIdx.x] * xs;
   }
}

template <typename T_ELEM, int blkSize, bool ISUNIT>
void __device__ dblkSolve_trans(const T_ELEM *minus_a, int lda, T_ELEM &val) {

   volatile T_ELEM __shared__ xs;

#pragma unroll 16
   for(int i=blkSize-1; i>=0; i--) {
      if(threadIdx.x==i) {
         if(!ISUNIT) val *= minus_a[i*lda+i];
         xs = val;
      }
      if(threadIdx.x < i)
         val += minus_a[i*lda+threadIdx.x] * xs;
   }
}

/* Copies a nbi x nbi block of a to provided cache.
 * Copies -a and only the half triangle
 */
template <typename T_ELEM, unsigned int nbi, unsigned int ntid, bool TRANS, bool ISUNIT, bool CONJG>
void __device__ tocache(unsigned int tid, const T_ELEM *a, int lda,
      T_ELEM *cache) {
   int x = tid % nbi;
   int y = tid / nbi;
   int ty = ntid/nbi;

   if(!TRANS) {
      for(int i=0; i<nbi; i+=ty) {
         if(x>(i+y)) cache[(i+y)*nbi+x] = CONJG ? cuConj(-a[(i+y)*lda+x]) : -a[(i+y)*lda+x];
         else if((i+y)<nbi) cache[(i+y)*nbi+x] = 0.0;
         if((!ISUNIT) && (x==(i+y)) ) cache[(i+y)*nbi+x] = 1 / (CONJG ? cuConj(a[(i+y)*lda+x]) : a[(i+y)*lda+x]);
      }
   } else {
      for(int i=0; i<nbi; i+=ty) {
         if(x>(i+y)) cache[(i+y)+nbi*x] = CONJG ? cuConj(-a[(i+y)*lda+x]) : -a[(i+y)*lda+x];
         else if((i+y)<nbi) cache[(i+y)+nbi*x] = 0.0;
         if((!ISUNIT) && (x==(i+y)) ) cache[(i+y)+nbi*x] = 1 / (CONJG ? cuConj(a[(i+y)*lda+x]) : a[(i+y)*lda+x]);
      }
   }
}

/* Copies an n x n block of a to provided cache, provided n<nbi.
 * If diag is true, then only copy lower triangle. 
 * ntid is the number of participating threads, tid is the thread id.
 */
template <typename T_ELEM, unsigned int nbi, unsigned int ntid, bool TRANS, bool ISUNIT, bool CONJG>
void __device__ tocache_small(int n, unsigned int tid, const T_ELEM *a,
      int lda, T_ELEM *cache) {
   int x = tid % nbi;
   int y = tid / nbi;
   int ty = ntid/nbi;
   if(!TRANS) {
      for(int i=0; i<n; i+=ty) {
         if(i+y>=nbi) continue; // past end of cache array
         if((i+y)<n && (x>(i+y) && x<n)) cache[(i+y)*nbi+x] = CONJG ? cuConj(-a[(i+y)*lda+x]) : -a[(i+y)*lda+x];
         else                            cache[(i+y)*nbi+x] = 0.0;
         if((!ISUNIT) && x==(i+y) && x<n) cache[(i+y)*nbi+x] = 1 / (CONJG ? cuConj(a[(i+y)*lda+x]) : a[(i+y)*lda+x]);
      }
   } else {
      for(int i=0; i<nbi; i+=ty) {
         if(i+y>=nbi) continue; // past end of cache array
         if((i+y)<n && x>(i+y) && x<n) cache[(i+y)+nbi*x] = CONJG ? cuConj(-a[(i+y)*lda+x]) : -a[(i+y)*lda+x];
         else                          cache[(i+y)+nbi*x] = 0.0;
         if((!ISUNIT) && x==(i+y) && x<n) cache[(i+y)+nbi*x] = 1 / (CONJG ? cuConj(a[(i+y)*lda+x]) : a[(i+y)*lda+x]);
      }
   }
}

/*
 * Copies a n x nbi block of a into rect
 */
template <typename T_ELEM, unsigned int nbi, unsigned int ntid>
void __device__ cacherect(unsigned int tid, int m, int n, const T_ELEM *a,
      int lda, T_ELEM *rect, int ldr) {
   if(m==0 || n==0) return;
   int mb = ((m-1)/nbi + 1) * nbi;
   int x = tid % mb;
   int y = tid / mb;
   int ty = ntid/mb;
   for(int i=0; i<n; i+=ty) {
      if(i+y<n && x<m)
         rect[(i+y)*ldr+x] = a[(i+y)*lda+x];
   }
}

/*
 * Copies a nbi x n block of a into rect
 */
template <typename T_ELEM, unsigned int nbi, unsigned int ntid>
void __device__ cacherect_trans(unsigned int tid, int m, int n, const T_ELEM *a,
      int lda, T_ELEM *rect, int ldr) {
   int x = tid % nbi;
   int y = tid / nbi;
   int ty = ntid/nbi;
   for(int i=0; i<n; i+=ty) {
      if(x<m && i+y<n)
         rect[(i+y)+x*ldr] = a[(i+y)*lda+x];
   }
}

/*
 * Perform a triangular solve on an n x n matrix, where n%blkSize==0.
 */
template <typename T_ELEM, unsigned int n, unsigned int blkSize,
   unsigned int threadsx, unsigned int threadsy,
   bool ISUNIT, bool CONJG>
__launch_bounds__(threadsx*threadsy, 1)
void __global__ trsv_fixed_ln(const T_ELEM *a, int lda, T_ELEM *xglobal,
      int incx) {

   volatile T_ELEM __shared__ xshared_actual[n];
   volatile T_ELEM *xshared = xshared_actual;
   T_ELEM __shared__ cache_even[blkSize*blkSize];
   T_ELEM __shared__ cache_odd[blkSize*blkSize];
   T_ELEM __shared__ rect[(n>blkSize)? (n-blkSize)*blkSize : 1];
   int tid = threadsx*threadIdx.y+threadIdx.x;

   /* Precache x */
   if(tid<n) xshared[tid] = xglobal[tid*incx];

   /* Precache first block */
   tocache <T_ELEM, blkSize,threadsx*threadsy, false, ISUNIT, CONJG>
      (threadsx*threadIdx.y+threadIdx.x, a, lda, cache_even);
   __syncthreads();

#pragma unroll
   for(int ii=0; ii<n; ii+=blkSize) {
      /* Preload entries for rectanular block */
      if(n>blkSize && threadIdx.y!=0)
         cacherect <T_ELEM, blkSize, threadsx*(threadsy-1)>
            (threadsx*(threadIdx.y-1)+threadIdx.x, n-ii-blkSize, blkSize,
             a+blkSize, lda, rect, n-blkSize);
      /* Solve diagonal block (one warp only) */
      if(tid<blkSize) {
         T_ELEM val = xshared[tid];
         if(ii%(2*blkSize)==0) {
            dblkSolve<T_ELEM,blkSize,ISUNIT>(cache_even, blkSize, val);
         } else {
            dblkSolve<T_ELEM,blkSize,ISUNIT>(cache_odd, blkSize, val);
         }
         xshared[tid] = val;
      }
      else if(n>blkSize && ii+blkSize<n) {
         if(ii%(2*blkSize)==0) {
            tocache <T_ELEM,blkSize,threadsx*threadsy-32,false,ISUNIT,CONJG>
               (threadsx*threadIdx.y+threadIdx.x-32, &a[blkSize*(lda+1)], lda,
                cache_odd);
         } else {
            tocache <T_ELEM,blkSize,threadsx*threadsy-32,false,ISUNIT,CONJG>
               (threadsx*threadIdx.y+threadIdx.x-32, &a[blkSize*(lda+1)], lda,
                cache_even);
         }
      }
      __syncthreads();
      /* Apply rectangular block (one thread per row) */
      if(n>blkSize && threadIdx.y==0 && ii+blkSize+threadIdx.x<n) {
         T_ELEM val=0;
         for(int i=0; i<blkSize; i++) {
            val += rect[(i)*(n-blkSize)+threadIdx.x] *
               xshared[i];
         }
         xshared[blkSize+threadIdx.x] -= val;
      }
      __syncthreads();
      a+=blkSize*(lda+1);
      xshared+=blkSize;
   }
   /* Store x back to global memory */
   if(tid<n) xglobal[tid*incx] = xshared_actual[tid];
}

/*
 * Perform a triangular solve on an n x n matrix, where n%blkSize==0.
 */
template <typename T_ELEM, unsigned int n, unsigned int blkSize,
   unsigned int threadsx, unsigned int threadsy,
   bool ISUNIT, bool CONJG>
__launch_bounds__(threadsx*threadsy, 1)
void __global__ trsv_fixed_lt(const T_ELEM *a, int lda, T_ELEM *xglobal,
      int incx) {

   volatile T_ELEM __shared__ xshared_actual[n];
   volatile T_ELEM *xshared = xshared_actual+n-blkSize;
   T_ELEM __shared__ cache_even[blkSize*blkSize];
   T_ELEM __shared__ cache_odd[blkSize*blkSize];
   T_ELEM *this_cache, *next_cache;
   T_ELEM __shared__ rect[(n>blkSize)? (n-blkSize)*blkSize : 1];
   int tid = threadsx*threadIdx.y+threadIdx.x;

   /* Precache x */
   if(tid<n) xshared_actual[tid] = xglobal[tid*incx];

   /* Precache first block */
   this_cache = cache_even;
   next_cache = cache_odd;
   tocache <T_ELEM,blkSize,threadsx*threadsy,true,ISUNIT,CONJG>
      (threadsx*threadIdx.y+threadIdx.x, &a[(n-blkSize)*(lda+1)], lda,
      this_cache);
   __syncthreads();


#pragma unroll
   for(int ii=n-blkSize; ii>=0; ii-=blkSize) {
      /* Preload entries for rectangular block */
      if(n>blkSize && threadIdx.y!=0)
         cacherect_trans <T_ELEM, blkSize, threadsx*(threadsy-1)>
            (threadsx*(threadIdx.y-1)+threadIdx.x, blkSize, ii, a+ii,
             lda, rect, n-blkSize);

      /* Solve diagonal block (one warp only) */
      if(tid<blkSize) {
         T_ELEM val = xshared[tid];
         dblkSolve_trans<T_ELEM,blkSize,ISUNIT>(this_cache, blkSize, val);
         xshared[tid] = val;
      }
      else if(n>blkSize && ii>=blkSize) {
         tocache <T_ELEM, blkSize,threadsx*threadsy-blkSize, true, ISUNIT, CONJG>
            (threadsx*threadIdx.y+threadIdx.x-blkSize, &a[(ii-blkSize)*(lda+1)],
             lda, next_cache);
      }
      __syncthreads();
      /* Apply rectangular block (one thread per row) */
      if(n>blkSize && threadIdx.y==0 && threadIdx.x<ii) {
         T_ELEM val=0;
         for(int i=0; i<blkSize; i++) {
            val += rect[(i)*(n-blkSize)+threadIdx.x] *
               xshared[i];
         }
         xshared_actual[threadIdx.x] -= val;
      }
      __syncthreads();
      //a+=blkSize*(lda+1);
      xshared-=blkSize;
      /* Switch caches for next iteration */
      if(this_cache==cache_even) {
         this_cache = cache_odd;
         next_cache = cache_even;
      } else {
         this_cache = cache_even;
         next_cache = cache_odd;
      }
   }
   /* Store x back to global memory */
   if(tid<n) xglobal[tid*incx] = xshared_actual[tid];

}

/*
 * Perform a triangular solve on an n x n matrix.
 * Allows n%blkSize!=0 (unlike trsv_fixed_ln), but pays a performance hit
 * for doing so.
 */
template <typename T_ELEM, int maxn, int blkSize,
   int threadsx, int threadsy,
   bool ISUNIT, bool CONJG>
__launch_bounds__(threadsx*threadsy, 1)
void __global__ trsv_variable_ln(int n, const T_ELEM *a, int lda,
      T_ELEM *xglobal, int incx) {

   volatile T_ELEM __shared__ xshared_actual[maxn];
   volatile T_ELEM *xshared = xshared_actual;
   T_ELEM __shared__ cache_even[blkSize*blkSize];
   T_ELEM __shared__ cache_odd[blkSize*blkSize];
   T_ELEM __shared__ rect[(maxn>blkSize)? (maxn-blkSize)*blkSize : 1];
   int tid = threadsx*threadIdx.y+threadIdx.x;
   T_ELEM *this_cache, *next_cache;

   /* Precache x */
   if(tid<n) xshared[tid] = xglobal[tid*incx];

   /* Precache first block */
   this_cache = cache_even;
   next_cache = cache_odd;
   if(n<blkSize) {
      tocache_small <T_ELEM, blkSize,threadsx*threadsy, false, ISUNIT, CONJG>
         (n, threadsx*threadIdx.y+threadIdx.x, a, lda, this_cache);
   } else {
      tocache <T_ELEM, blkSize,threadsx*threadsy, false, ISUNIT, CONJG>
         (threadsx*threadIdx.y+threadIdx.x, a, lda, this_cache);
   }
   __syncthreads();

   for(int ii=0; ii<=n-blkSize; ii+=blkSize) {
      /* Preload entries for rectanular block */
      if(threadIdx.y!=0)
         cacherect <T_ELEM, blkSize, threadsx*(threadsy-1)>
            (threadsx*(threadIdx.y-1)+threadIdx.x, n-ii-blkSize, blkSize,
             a+blkSize, lda, rect, n-blkSize);
      /* Solve diagonal block (one warp only) */
      if(tid<blkSize) {
         T_ELEM val = xshared[tid];
         dblkSolve<T_ELEM,blkSize,ISUNIT>(this_cache, blkSize, val);
         xshared[tid] = val;
      }
      else if(ii+2*blkSize<=n)
         tocache <T_ELEM, blkSize,threadsx*threadsy-32, false, ISUNIT, CONJG>
            (threadsx*threadIdx.y+threadIdx.x-32, &a[blkSize*(lda+1)], lda,
             next_cache);
      else if(ii+blkSize<n)
         tocache_small <T_ELEM, blkSize,threadsx*threadsy-32, false, ISUNIT, CONJG>
            (n-(ii+blkSize), threadsx*threadIdx.y+threadIdx.x-32,
             &a[blkSize*(lda+1)], lda, next_cache);
      __syncthreads();
      /* Apply rectangular block (one thread per row) */
      if(threadIdx.y==0 && ii+blkSize+threadIdx.x<n) {
         T_ELEM val=0;
         for(int i=0; i<blkSize; i++) {
            val += rect[(i)*(n-blkSize)+threadIdx.x] *
               xshared[i];
         }
         xshared[blkSize+threadIdx.x] -= val;
      }
      __syncthreads();
      a+=blkSize*(lda+1);
      xshared+=blkSize;
      /* Swap caches */
      T_ELEM *temp = this_cache;
      this_cache = next_cache;
      next_cache = temp;
   }

   /* Final block */
   if(threadIdx.y==0 && threadIdx.x<blkSize) {
      volatile T_ELEM __shared__ xs;
      T_ELEM val = xshared[threadIdx.x];
      for(int i=0; i<n%blkSize; i++) {
         if(threadIdx.x==i) {
            if(!ISUNIT) val *= this_cache[i*blkSize+i];
            xs = val;
         }
         if(threadIdx.x>=i+1)
            val += this_cache[i*blkSize+threadIdx.x] * xs;
      }
      xshared[threadIdx.x] = val;
   }
   __syncthreads();

   /* Store x back to global memory */
   if(tid<n) xglobal[tid*incx] = xshared_actual[tid];
}

/*
 * Perform a triangular solve on an n x n matrix.
 * Allows n%blkSize!=0. (unlike trsv_fixed_lt), but pays a performanc hit
 * for doing so.
 */
template <typename T_ELEM, int maxn, int blkSize, int threadsx, int threadsy,
   bool ISUNIT, bool CONJG>
__launch_bounds__(threadsx*threadsy, 1)
void __global__ trsv_variable_lt(int n, const T_ELEM *a, int lda,
      T_ELEM *xglobal, int incx) {

   volatile T_ELEM __shared__ xshared_actual[maxn];
   volatile T_ELEM *xshared = xshared_actual+((n-1)/blkSize)*blkSize;
   T_ELEM __shared__ cache_even[blkSize*blkSize];
   T_ELEM __shared__ cache_odd[blkSize*blkSize];
   T_ELEM *this_cache, *next_cache;
   T_ELEM __shared__ rect[(maxn>blkSize)? (maxn-blkSize)*blkSize : 1];
   int tid = threadsx*threadIdx.y+threadIdx.x;

   /* Precache x */
   if(tid<n) xshared_actual[tid] = xglobal[tid*incx];
   if(tid>=n && tid<maxn) xshared_actual[tid] = 0;

   /* Precache first block */
   this_cache = cache_even;
   next_cache = cache_odd;
   tocache_small <T_ELEM, blkSize,threadsx*threadsy, true, ISUNIT, CONJG>
      ((n-1)%blkSize+1, threadsx*threadIdx.y+threadIdx.x, &a[(((n-1)/blkSize)*blkSize)*(lda+1)], lda,
      this_cache);
   __syncthreads();

   for(int ii=((n-1)/blkSize)*blkSize; ii>=0; ii-=blkSize) {
      /* Preload entries for rectangular block */
      if(n>blkSize && threadIdx.y!=0)
         cacherect_trans <T_ELEM, blkSize, threadsx*(threadsy-1)>
            (threadsx*(threadIdx.y-1)+threadIdx.x, MIN(n-ii,blkSize), ii, a+ii,
             lda, rect, ii);

      /* Solve diagonal block (one warp only) */
      if(tid<blkSize) {
         T_ELEM val = xshared[tid];
         dblkSolve_trans<T_ELEM,blkSize,ISUNIT>(this_cache, blkSize, val);
         xshared[tid] = val;
      }
      else if(n>blkSize && ii>=blkSize) {
         tocache <T_ELEM, blkSize,threadsx*threadsy-blkSize, true, ISUNIT, CONJG>
            (threadsx*threadIdx.y+threadIdx.x-blkSize, &a[(ii-blkSize)*(lda+1)],
             lda, next_cache);
      }
      __syncthreads();
      /* Apply rectangular block (one thread per row) */
      if(n>blkSize && threadIdx.y==0 && threadIdx.x<ii) {
         T_ELEM val=0;
         for(int i=0; i<MIN(n-ii,blkSize); i++)
            val += rect[i*ii+threadIdx.x] * xshared[i];
         xshared_actual[threadIdx.x] -= val;
      }
      __syncthreads();
      //a+=blkSize*(lda+1);
      xshared-=blkSize;
      /* Switch caches for next iteration */
      if(this_cache==cache_even) {
         this_cache = cache_odd;
         next_cache = cache_even;
      } else {
         this_cache = cache_even;
         next_cache = cache_odd;
      }
   }
   /* Store x back to global memory */
   if(tid<n) xglobal[tid*incx] = xshared_actual[tid];

}


/* loops until *sync > val.
 * Needs to be seperate function to force volatile onto *sync.
 */
void __device__ wait_until_ge(int tid, volatile int *sync, int col_to_wait, int *col_done) {
   if(tid == 0) {
      /* Only read global memory when necessary */
      if (*col_done < col_to_wait) {
         while(*sync < col_to_wait) {}
         *col_done = *sync;
      }
   }
   __syncthreads();
}

/* Returns next block row index that requires processing */
int __device__ nextRow(int *address) {
   volatile int __shared__ old;
   if(threadIdx.x==0 && threadIdx.y==0)
      old = atomicAdd(address, 1);
   __syncthreads();
   return old;
}

/*
   Solves the system
      L_22 X_21 = - L_21 X_11
   for X_21.
*/
template <typename T_ELEM, int n, int lda, int threadsx, int threadsy, bool ISUNIT>
void __device__ slv21(const T_ELEM *x11, T_ELEM *a21, const T_ELEM *l22, volatile T_ELEM *xsarray) {

   const int tid = threadsx*threadIdx.y+threadIdx.x;
   const int ntid = threadsx*threadsy;
   const int x = (n>0) ? tid % n : 0;
   const int y = (n>0) ? tid / n : 0;
   const int ty = (n>0) ? ntid/n : 1;

   /* Note: as different threads within a warp can work on different
      columns, we need different xs variables (one per col being worked on) */
   volatile T_ELEM *xs = &xsarray[y];

   if(y>n) return;

#pragma unroll
   for(int j=0; j<n; j+=ty) {
      if(j+y>=n) continue;

      /* construct col (j+y) of -L_21 X_11 */
      T_ELEM val = 0;
      for(int k=j; k<n; k++) {
         if(k+y<n) val += a21[(k+y)*lda+x] * x11[(j+y)*lda+k+y];
      }
      val = -val;

      /* solve L_22 X_21(col j) = a21(col j) in place */
#pragma unroll 2
      for(int k=0; k<n; k++) { // Column of l22, must be done in order
         if(x==k) {
            if(!ISUNIT) val *= l22[k*lda+k];
            xs[0] = val;
         }
         if(x>k)
            val += l22[k*lda+x]*xs[0];
      }
      a21[(j+y)*lda+x] = -val;
   }
}

/* Take transpose of a matrix in shared memory */
template <typename T_ELEM, int threadsy, int lda>
void __device__ transpose(int n, const T_ELEM *a, T_ELEM *at) {
   if(threadIdx.y==0 && threadIdx.x<n) {
      for(int j=0; j<n; j++)
         at[j*lda+threadIdx.x] = a[threadIdx.x*lda+j];
   }
}

/* Invert a lower triangular matrix recursively using formula
 * ( L_11      ) ^-1 = ( L_11^-1                        )
 * ( L_21 L_22 )       ( -L_22^-1*L_21*L_11^-1  L22^_-1 )
 *
 * Note: Expects -L to be passed in, and factorises to +L
 *
 * (This method is recommended as componentwise backwards stable for
 * divide an conquer computation of triangular matrix in version in:
 * Stability of parallel triangular system solvers, Higham, 1995)
 */
template <typename T_ELEM, int n, int lda, int threadsx, int threadsy, bool ISUNIT, bool TRANS>
void __device__ invert(T_ELEM *a, volatile T_ELEM /*__shared__*/ *xsarray) {

   if(n==2) {
      if(threadIdx.x==0 && threadIdx.y==0) {
         if(ISUNIT) {
            a[0] = 1;
            a[lda+1] = 1;
            a[1] = a[1];
         } else {
            a[0] = a[0];
            a[lda+1] = a[lda+1];
            a[1] = a[1]*(a[0]*a[lda+1]);
         }
         if(TRANS) a[lda] = a[1];
      }
   } else {
      invert<T_ELEM, n/2, lda, threadsx, threadsy, ISUNIT, TRANS>(a, xsarray); // A_11
      __syncthreads();
      slv21<T_ELEM, n/2, lda, threadsx, threadsy, ISUNIT>(a, &a[n/2], &a[(lda+1)*n/2], xsarray); // A_21
      if(TRANS) {
         __syncthreads();
         transpose<T_ELEM, threadsy, lda> (n/2, &a[n/2], &a[(n/2)*lda]);
      }
      __syncthreads();
      invert<T_ELEM, n/2, lda, threadsx, threadsy, ISUNIT, TRANS>(&a[(lda+1)*n/2], xsarray); // A_22
   }
}

/* 
 * Performs a solve through a precalulated matrix inverse
 * (so actually a triangular matrix-vector multiply)
 */
template<typename T_ELEM, int n, int threadsy>
void __device__ slvinv(const T_ELEM *a, T_ELEM *xshared, T_ELEM &val,
      T_ELEM *partSum) {

   a += threadIdx.y*n+threadIdx.x;
   xshared += threadIdx.y;

   if(threadIdx.y==0) {
      xshared[threadIdx.x] = val;
   }
   __syncthreads();

   /* matrix-vector multiply for solution */
   if(threadIdx.y<threadsy && threadIdx.x<n) {
      val=0;
      for(int j=0; j<n; j+=threadsy) {
         val += a[j*n] * xshared[j];
      }
      partSum[threadIdx.y*n+threadIdx.x] = val;
   }
   __syncthreads();
   if(threadIdx.y==0) {
      for(int i=1; i<threadsy; i++)
         val += partSum[i*n+threadIdx.x];
   }
}

/* 
 * Performs a solve through a transpose precalulated matrix inverse
 * (so actually a transpose triangular matrix-vector multiply)
 */
template<typename T_ELEM, int n, int threadsy>
void __device__ slvinv_trans(const T_ELEM *a, T_ELEM *xshared, T_ELEM &val,
      T_ELEM *partSum, int row) {

   a += threadIdx.y*n+threadIdx.x;
   xshared += threadIdx.y;

   if(threadIdx.y==0) {
      xshared[threadIdx.x] = val;
   }
   __syncthreads();

   /* matrix-vector multiply for solution */
   val=0;
   if(threadIdx.x<n) {
      for(int j=0; j<n; j+=threadsy) {
         if(threadIdx.x <= j+threadIdx.y) {
            val += a[j*n] * xshared[j];
         }
      }
   }
   partSum[threadIdx.y*n+threadIdx.x] = val;
   __syncthreads();
   if(threadIdx.y==0) {
      for(int i=1; i<threadsy; i++)
         val += partSum[i*n+threadIdx.x];
   }
}


/* Sets sync values correctly prior to call to trsv_ln_exec */
void __global__ trsv_init(int *sync) {
   sync[0] = -1; // Last ready column
   sync[1] = 0; // Next row to assign
}

/* Performs trsv for Non-transposed Lower-triangular matrices
 * Requires trsv_init() to be called first to initialize sync[].
 */
template <typename T_ELEM, unsigned int nb, unsigned int threadsx,
   unsigned int threadsy, bool ISUNIT, bool CONJG>
__launch_bounds__(threadsx*threadsy, 4)
/* Note: setting above occupany to 5 causes random errors on large problems:
   suspect compiler bug */
void __global__ trsv_ln_exec(int n, const T_ELEM* __restrict__ a, int lda,
      T_ELEM* __restrict__ xglobal, int incx, int* __restrict__ sync
#ifdef TIMING
      , unsigned int* __restrict__ prof
#endif /* TIMING */
      ) {

   int tid = threadsx*threadIdx.y + threadIdx.x;

   /* sync components:
    *    sync[0] => Last ready column [init to -1]
    *    sync[1] => Next row to assign [init to 0]
    */

   T_ELEM __shared__ partSum[threadsy*threadsx];
   T_ELEM __shared__ cache[nb*nb];
   T_ELEM __shared__ xlocal[nb];
   T_ELEM regcache[nb/threadsy];

   if(incx<0) xglobal+=(1-n)*incx;

   /* Get row handled by this block */
   int row = nextRow(&sync[1]);

#ifdef TIMING
   /* prof[2*(nblk+1)*(nblk+1) + row]  SM of block row
    * prof[2*((nblk+1)*row + col)]     point to length two sa, en array.
    *                                  init is treated as col=nblk.
    */
   int nblk = n / nb;
   unsigned int tsa, ten;
   if(tid==0) {
      prof[2*(nblk+1)*(nblk+1)+row] = getSM();
      prof = &prof[2*(nblk+1)*row]; // Move to start of array for this block row
      prof[2*nblk+0] = clock(); // start of init
   }
#endif

   bool short_row = ((n-1)/nb==row && n%nb!=0); /* requires special handling */

   if(row!=0) {
      const T_ELEM *aval = &a[((row-1)*nb+threadIdx.y)*lda+row*nb+threadIdx.x];
#pragma unroll
      for(int j=0; j<nb; j+=threadsy)
         regcache[j/threadsy] = aval[j*lda];
   }

   /* Copy diagonal block to shared memory */
   if(!short_row) {
      /* on a block row of full size */
      tocache <T_ELEM,nb,threadsx*threadsy,false,ISUNIT,CONJG> (tid, &a[row*nb*lda+row*nb], lda, cache);
   } else {
      /* on last row, smaller than full blkSize */
      tocache_small <T_ELEM,nb,threadsx*threadsy,false,ISUNIT,CONJG> (n%nb, tid, &a[row*nb*lda+row*nb], lda, cache);
   }
   __syncthreads();

#ifdef INV_AFTER
      if(row>=INV_AFTER)
         invert<T_ELEM, nb, nb, threadsx, threadsy, ISUNIT, false>(cache, partSum);
#endif /* INV_AFTER */

#ifdef TIMING
   __syncthreads();
   if(tid==0) prof[2*nblk+1] = clock(); // end of init
#endif

   /* Loop over blocks as they become available */
   T_ELEM val = 0;
   if(threadIdx.y==0) {
      if(!short_row) {
         val = -xglobal[int(row*nb+threadIdx.x)*incx];
      } else {
         if(threadIdx.x<n%nb) val = -xglobal[int(row*nb+threadIdx.x)*incx];
      }
   }
   int col_done = -1;
   for(int col=0; col<row-1; col++) {
      /* apply update from block (row, col) */
      const T_ELEM *aval = &a[(col*nb+threadIdx.y)*lda + row*nb+threadIdx.x];
      T_ELEM *xg = &xglobal[int(col*nb)*incx];
      wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
#ifdef TIMING
      tsa = clock(); // record start of block (row,col)
#endif // TIMING
      T_ELEM *xl = xlocal+threadIdx.y;
      if(tid<nb) xlocal[tid] = loadVolatile(&xg[int(tid)*incx]);
      __syncthreads();
      if(nb % threadsy == 0) {
#pragma unroll
         for(int j=0; j<nb; j+=threadsy)
            val += aval[j*lda] * xl[j];
      } else {
#pragma unroll
         for(int j=0; j<nb; j+=threadsy)
            if(j+threadIdx.y<nb) val += aval[j*lda] * xl[j];
      }
#ifdef TIMING
      if(tid==0) {
         ten = clock(); // record end of block (row,col)
         prof[2*col+0] = tsa; // start of block (row,col)
         prof[2*col+1] = ten; // end of block (row,col)
      }
#endif // TIMING
   }
   if(row!=0) {
      const int col = row-1;
      /* apply update from block (row, col) */
      T_ELEM *xg = &xglobal[int(col*nb)*incx];
      wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
#ifdef TIMING
      tsa = clock(); //start of block (row-1,row)
#endif // TIMING
      T_ELEM *xl = xlocal+threadIdx.y;
      if(tid<nb) xlocal[tid] = loadVolatile(&xg[int(tid)*incx]);
      __syncthreads();
#pragma unroll
      for(int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
         val += regcache[j/threadsy] * xl[j];
#ifdef TIMING
      if(tid==0) {
         ten = clock(); // end of block
         prof[2*col+0] = tsa;
         prof[2*col+1] = ten;
      }
#endif // TIMING
   }
   partSum[threadIdx.y*threadsx+threadIdx.x] = val;
   __syncthreads();
   if(threadIdx.y==0) {
      for(int i=1; i<threadsy; i++)
         val += partSum[i*threadsx+threadIdx.x];
      val = -val;
      if(short_row && threadIdx.x>=n%nb) val = 0.0;
   }

   /* Apply update from diagonal block (row, row) */
#ifdef TIMING
   __syncthreads();
   tsa = clock(); // start of diagonal block
#endif // TIMING
#ifdef INV_AFTER
      if(row>=INV_AFTER) {
         slvinv<T_ELEM, nb, threadsy>(cache, xlocal, val, partSum);
         if(!short_row || threadIdx.x<n%nb) {
            if(threadIdx.y==0) {
               xglobal[int(row*nb+tid)*incx] = val;
            }
         }
      } else {
         if(threadIdx.y==0) {
            dblkSolve<T_ELEM,nb,ISUNIT>(cache, nb, val);
            if(!short_row || threadIdx.x<n%nb) {
               xglobal[int(row*nb+tid)*incx] = val;
            }
         }
      }
#else /* INV_AFTER */
      if(threadIdx.y==0) {
         dblkSolve<T_ELEM,nb,ISUNIT>(cache, nb, val);
         if(!short_row || threadIdx.x<n%nb) {
            xglobal[int(row*nb+tid)*incx] = val;
         }
      }
#endif /* INV_AFTER */
#ifdef TIMING
   if(tid==0) {
      ten = clock(); // end of diagonal block
      prof[2*row+0] = tsa;
      prof[2*row+1] = ten;
   }
#endif // TIMING
   /* Notify other blocks that soln is ready for this row */
   __threadfence(); // Wait for xglobal to be visible to other blocks
   if(tid==0) atomicAdd(&sync[0],1); // Use atomicAdd to bypass L1 miss
   __threadfence(); // Flush sync[0] asap
}

/* Performs trsv for Transposed Lower-triangular matrices
 * Requires trsv_init() to be called first to initialize sync[].
 */
template <typename T_ELEM, unsigned int nb, unsigned int threadsx,
   unsigned int threadsy, bool ISUNIT, bool CONJG>
__launch_bounds__(threadsx*threadsy, 4)
void __global__ trsv_lt_exec(int n, const T_ELEM *a, int lda,
      T_ELEM *xglobal, int incx, int *sync) {

   int nblk = (n-1) / nb + 1;
   int tid = threadsx*threadIdx.y + threadIdx.x;

   /* sync components:
    *    sync[0] => nblk - Last ready column [init to -1]
    *    sync[1] => nblk - Next row to assign [init to 0]
    */

   T_ELEM __shared__ partSum[threadsy*threadsx];
   T_ELEM __shared__ cache[nb*nb];
   T_ELEM regcache[nb/threadsy];
   T_ELEM ps[nb/threadsy];
   if(incx<0) xglobal+=(n-1)*(-incx); // if incx negative, start at end

   /* Get row handled by this block */
   int row = nblk-1 - nextRow(&sync[1]);

   bool short_row = ((n-1)/nb==row && n%nb!=0); /* requires special handling */

   if(row!=nblk-1) {
      const T_ELEM *aval = &a[(row*nb+threadIdx.x)*lda+(row+1)*nb+threadIdx.y];
#pragma unroll
      for(int j=0; j<nb; j+=threadsy)
         regcache[j/threadsy] = aval[j];
   }

   /* Copy diagonal block to shared memory */
   if(!short_row) {
      /* on a block row of full size */
#ifdef INV_AFTER
      if(nblk-1-row>=INV_AFTER) {
         tocache <T_ELEM,nb,threadsx*threadsy,false,ISUNIT,CONJG> (tid, &a[row*nb*lda+row*nb], lda, cache);
      } else {
         tocache <T_ELEM,nb,threadsx*threadsy,true,ISUNIT,CONJG> (tid, &a[row*nb*lda+row*nb], lda, cache);
      }
#else /* INV_AFTER */
      tocache <T_ELEM,nb,threadsx*threadsy,true,ISUNIT,CONJG> (tid, &a[row*nb*lda+row*nb], lda, cache);
#endif /* INV_AFTER */
   } else {
      /* on last row, smaller than full blkSize */
#ifdef INV_AFTER
      if(nblk-1-row>=INV_AFTER) {
         tocache_small <T_ELEM,nb,threadsx*threadsy,false,ISUNIT,CONJG> (n%nb, tid, &a[row*nb*lda+row*nb], lda, cache);
      } else {
         tocache_small <T_ELEM,nb,threadsx*threadsy,true,ISUNIT,CONJG> (n%nb, tid, &a[row*nb*lda+row*nb], lda, cache);
      }
#else /* INV_AFTER */
      tocache_small <T_ELEM,nb,threadsx*threadsy,true,ISUNIT,CONJG> (n%nb, tid, &a[row*nb*lda+row*nb], lda, cache);
#endif /* INV_AFTER */
   }
   __syncthreads();

#ifdef INV_AFTER
   if(nblk-1-row>=INV_AFTER)
      invert<T_ELEM, nb, nb, threadsx, threadsy, ISUNIT, true>(cache, partSum);
#endif /* INV_AFTER */

   /* Loop over blocks as they become available */
   volatile T_ELEM __shared__ soln[nb];
   if(threadIdx.y==0) {
      if(!short_row) {
         soln[threadIdx.x] = xglobal[int(row*nb+threadIdx.x)*incx];
      } else {
         if(threadIdx.x<n%nb)
            soln[threadIdx.x] = xglobal[int(row*nb+threadIdx.x)*incx];
         else 
            soln[threadIdx.x] = 0;
      }
   }
#pragma unroll
   for(int j=0; j<nb/threadsy; j++) ps[j] = 0;
   int col_done = -1;
   for(int col=nblk-1; col>row+1; col--) {
      /* apply update from block (row, col) */
      const T_ELEM *aval = &a[(row*nb+threadIdx.y)*lda + col*nb+threadIdx.x];
      T_ELEM *xg = &xglobal[int(col*nb)*incx];
      wait_until_ge(tid, &sync[0], nblk-1-col, &col_done); // Wait for diagonal block to be done
      T_ELEM xl;
      if(col<nblk-1) {
         xl = loadVolatile(&xg[int(threadIdx.x)*incx]);
      } else {
         if(threadIdx.x<(n-1)%nb+1) xl = loadVolatile(&xg[int(threadIdx.x)*incx]);
         else                       xl = 0;
      }
      if(nb % threadsy == 0) {
        if(col!=nblk-1 || n%nb==0) {
#pragma unroll
            for(int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
               ps[j/threadsy] += aval[j*lda] * xl;
        } else {
            for(int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
               if(threadIdx.x<n%nb) ps[j/threadsy] += aval[j*lda] * xl;
        }
      } else {
#pragma unroll
         for(int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
            if(j+threadIdx.y<nb)
               ps[j/threadsy] += aval[j*lda] * xl;
      }
   }
   T_ELEM val = 0;
#pragma unroll
   for(int i=0; i<nb; i+=threadsy) {
      partSum[threadIdx.x*threadsy+threadIdx.y] = ps[i/threadsy];
      __syncthreads();
      if(threadIdx.y==0 && threadIdx.x>=i && threadIdx.x<i+threadsy) {
         for(int j=0; j<nb; j++)
            val += partSum[(threadIdx.x-i)+threadsy*j];
      }
      __syncthreads();
   }
   if(row!=nblk-1) {
      /* apply update from block (row, col) */
      const int col = row+1;
      T_ELEM *xg = &xglobal[int(col*nb)*incx];
      wait_until_ge(tid, &sync[0], nblk-1-col, &col_done); // Wait for diagonal block to be done
      T_ELEM __shared__ xlocal[nb];
      T_ELEM *xl = xlocal+threadIdx.y;
      if(col<nblk-1) {
         if(tid<nb) xlocal[tid] = loadVolatile(&xg[int(tid)*incx]);
         __syncthreads();
#pragma unroll
         for(int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
            val += regcache[j/threadsy] * xl[j];
      } else {
         if(tid<(n-1)%nb+1) xlocal[tid] = loadVolatile(&xg[int(tid)*incx]);
         __syncthreads();
#pragma unroll
         for(int j=0; j<(n-1)%nb+1; j+=threadsy) // do j=0,nb-1,threadsy
            if(j+threadIdx.y<(n-1)%nb+1) val += regcache[j/threadsy] * xl[j];
      }
   }
   partSum[threadIdx.y*threadsx+threadIdx.x] = val;
   __syncthreads();
   if(threadIdx.y==0) {
      for(int i=1; i<threadsy; i++)
         val += partSum[i*threadsx+threadIdx.x];
      val = soln[threadIdx.x]-val;
   }

   /* Apply update from diagonal block (row, row) */
#ifdef INV_AFTER
      if(nblk-1-row>=INV_AFTER) {
         T_ELEM __shared__ xshared[nb];
         slvinv_trans<T_ELEM, nb, threadsy>(cache, xshared, val, partSum, row);
         if(!short_row || threadIdx.x<n%nb) {
            if(threadIdx.y==0) {
               xglobal[int(row*nb+tid)*incx] = val;
            }
         }
      } else {
         if(threadIdx.y==0) {
            dblkSolve_trans<T_ELEM,nb,ISUNIT>(cache, nb, val);
            if(!short_row || threadIdx.x<n%nb) {
               xglobal[int(row*nb+tid)*incx] = val;
            }
         }
      }
#else /* INV_AFTER */
      if(threadIdx.y==0) {
         dblkSolve_trans<T_ELEM,nb,ISUNIT>(cache, nb, val);
         if(!short_row || threadIdx.x<n%nb) {
            xglobal[int(row*nb+tid)*incx] = val;
         }
      }
#endif /* INV_AFTER */
   /* Notify other blocks that soln is ready for this row */
   __threadfence(); // Wait for xglobal to be visible to other blocks
   if(tid==0) atomicAdd(&sync[0],1); // Use atomicAdd to bypass L1 miss
   __threadfence(); // Flush sync[0] asap
}

/* Copies a nbi x nbi block of a to provided cache.
 * Copies -a and only the half triangle
 */
template <typename T_ELEM, unsigned int nbi, unsigned int ntid, bool TRANS, bool ISUNIT, bool CONJG>
void __device__ tocache_upr(unsigned int tid, const T_ELEM *a, int lda,
      T_ELEM *cache) {
   int x = tid % nbi;
   int y = tid / nbi;
   int ty = ntid/nbi;

   if(!TRANS) {
      for(int i=0; i<nbi; i+=ty) {
         if((i+y)>=nbi) continue;
         if(x<(i+y)) cache[(i+y)*nbi+x] = CONJG ? cuConj(-a[(i+y)*lda+x]) : -a[(i+y)*lda+x];
         else cache[(i+y)*nbi+x] = 0.0;
         if( !ISUNIT && (x==i+y) )
            cache[(i+y)*nbi+x] = 1 / (CONJG ? cuConj(a[(i+y)*lda+x]) : a[(i+y)*lda+x]);
      }
   } else {
      for(int i=0; i<nbi; i+=ty) {
         if((i+y)>=nbi) continue;
         if(x<(i+y)) cache[(i+y)+nbi*x] = CONJG ? cuConj(-a[(i+y)*lda+x]) : -a[(i+y)*lda+x];
         else cache[(i+y)+nbi*x] = 0.0;
         if( !ISUNIT && (x==i+y) )
            cache[(i+y)+nbi*x] = 1 / (CONJG ? cuConj(a[(i+y)*lda+x]) : a[(i+y)*lda+x]);
      }
   }
}

/* Copies an n x n block of a to provided cache, provided n<nbi.
 * If diag is true, then only copy lower triangle. 
 * ntid is the number of participating threads, tid is the thread id.
 */
template <typename T_ELEM, unsigned int nbi, unsigned int ntid, bool TRANS, bool ISUNIT, bool CONJG>
void __device__ tocache_small_upr(int n, unsigned int tid, const T_ELEM *a,
      int lda, T_ELEM *cache) {
   int x = tid % nbi;
   int y = tid / nbi;
   int ty = ntid/nbi;

   if(!TRANS) {
      for(int i=0; i<nbi; i+=ty) {
         if((i+y)>=nbi) continue; // Past end of cache array
         if((i+y)<n && x<(i+y)) cache[(i+y)*nbi+x] = CONJG ? cuConj(-a[(i+y)*lda+x]) : -a[(i+y)*lda+x];
         else                   cache[(i+y)*nbi+x] = 0.0;
         if(!ISUNIT && x==(i+y)) {
            if(x<n) cache[(i+y)*nbi+x] = 1 / (CONJG ? cuConj(a[(i+y)*lda+x]) : a[(i+y)*lda+x]);
            else    cache[(i+y)*nbi+x] = 1.0;
         }
      }
   } else {
      for(int i=0; i<nbi; i+=ty) {
         if((i+y)>=nbi) continue; // Past end of cache array
         if((i+y)<n && x<(i+y)) cache[(i+y)+nbi*x] = CONJG ? cuConj(-a[(i+y)*lda+x]) : -a[(i+y)*lda+x];
         else                   cache[(i+y)+nbi*x] = 0.0;
         if(!ISUNIT && x==(i+y)) {
            if(x<n) cache[(i+y)+nbi*x] = 1 / (CONJG ? cuConj(a[(i+y)*lda+x]) : a[(i+y)*lda+x]);
            else    cache[(i+y)+nbi*x] = 1.0;
         }
      }
   }
}

/*
 * Perform a triangular solve on an n x n matrix, where n%blkSize==0.
 */
template <typename T_ELEM, unsigned int n, unsigned int blkSize,
   unsigned int threadsx, unsigned int threadsy,
   bool ISUNIT, bool CONJG>
__launch_bounds__(threadsx*threadsy, 1)
void __global__ trsv_fixed_un(const T_ELEM *a, int lda, T_ELEM *xglobal,
      int incx) {

   volatile T_ELEM __shared__ xshared_actual[n];
   volatile T_ELEM *xshared = xshared_actual+n-blkSize;
   T_ELEM __shared__ cache_even[blkSize*blkSize];
   T_ELEM __shared__ cache_odd[blkSize*blkSize];
   T_ELEM *this_cache, *next_cache;
   T_ELEM __shared__ rect[(n>blkSize)? (n-blkSize)*blkSize : 1];
   int tid = threadsx*threadIdx.y+threadIdx.x;

   /* Precache x */
   if(tid<n) xshared_actual[tid] = xglobal[tid*incx];

   /* Precache first block */
   this_cache = cache_even;
   next_cache = cache_odd;
   tocache_upr <T_ELEM,blkSize,threadsx*threadsy,false,ISUNIT,CONJG>
      (threadsx*threadIdx.y+threadIdx.x, &a[(n-blkSize)*(lda+1)], lda,
      this_cache);
   __syncthreads();

#pragma unroll
   for(int ii=n-blkSize; ii>=0; ii-=blkSize) {
      /* Preload entries for rectangular block */
      if(n>blkSize && threadIdx.y!=0)
         cacherect <T_ELEM, blkSize, threadsx*(threadsy-1)>
            (threadsx*(threadIdx.y-1)+threadIdx.x, ii, blkSize, a+ii*lda,
             lda, rect, n-blkSize);

      /* Solve diagonal block (one warp only) */
      if(tid<blkSize) {
         T_ELEM val = xshared[tid];
         dblkSolve_trans<T_ELEM,blkSize,ISUNIT>(this_cache, blkSize, val);
         xshared[tid] = val;
      }
      else if(n>blkSize && ii>=blkSize) {
         tocache_upr <T_ELEM, blkSize,threadsx*threadsy-blkSize, false, ISUNIT, CONJG>
            (threadsx*threadIdx.y+threadIdx.x-blkSize, &a[(ii-blkSize)*(lda+1)],
             lda, next_cache);
      }
      __syncthreads();
      /* Apply rectangular block (one thread per row) */
      if(n>blkSize && threadIdx.y==0 && threadIdx.x<ii) {
         T_ELEM val=0;
         for(int i=0; i<blkSize; i++) {
            val += rect[(i)*(n-blkSize)+threadIdx.x] *
               xshared[i];
         }
         xshared_actual[threadIdx.x] -= val;
      }
      __syncthreads();
      //a+=blkSize*(lda+1);
      xshared-=blkSize;
      /* Switch caches for next iteration */
      if(this_cache==cache_even) {
         this_cache = cache_odd;
         next_cache = cache_even;
      } else {
         this_cache = cache_even;
         next_cache = cache_odd;
      }
   }
   /* Store x back to global memory */
   if(tid<n) xglobal[tid*incx] = xshared_actual[tid];

}

/*
 * Perform a triangular solve on an n x n matrix.
 * Allows n%blkSize!=0. (unlike trsv_fixed_lt), but pays a performanc hit
 * for doing so.
 */
template <typename T_ELEM, int maxn, int blkSize, int threadsx, int threadsy,
   bool ISUNIT, bool CONJG>
__launch_bounds__(threadsx*threadsy, 1)
void __global__ trsv_variable_un(int n, const T_ELEM *a, int lda,
      T_ELEM *xglobal, int incx) {

   volatile T_ELEM __shared__ xshared_actual[maxn];
   volatile T_ELEM *xshared = xshared_actual+((n-1)/blkSize)*blkSize;
   T_ELEM __shared__ cache_even[blkSize*blkSize];
   T_ELEM __shared__ cache_odd[blkSize*blkSize];
   T_ELEM *this_cache, *next_cache;
   T_ELEM __shared__ rect[(maxn>blkSize)? (maxn-blkSize)*blkSize : 1];
   int tid = threadsx*threadIdx.y+threadIdx.x;

   /* Precache x */
   if(tid<n) xshared_actual[tid] = xglobal[tid*incx];
   if(tid>=n && tid<maxn) xshared_actual[tid] = 0;

   /* Precache first block */
   this_cache = cache_even;
   next_cache = cache_odd;
   tocache_small_upr <T_ELEM, blkSize,threadsx*threadsy, false, ISUNIT, CONJG>
      ((n-1)%blkSize+1, threadsx*threadIdx.y+threadIdx.x, &a[(((n-1)/blkSize)*blkSize)*(lda+1)], lda,
      this_cache);
   __syncthreads();

   for(int ii=((n-1)/blkSize)*blkSize; ii>=0; ii-=blkSize) {
      /* Preload entries for rectangular block */
      if(n>blkSize && threadIdx.y!=0)
         cacherect<T_ELEM, blkSize, threadsx*(threadsy-1)>
            (threadsx*(threadIdx.y-1)+threadIdx.x, ii, MIN(n-ii,blkSize),
             a+ii*lda, lda, rect, ii);

      /* Solve diagonal block (one warp only) */
      if(tid<blkSize) {
         T_ELEM val = xshared[tid];
         dblkSolve_trans<T_ELEM,blkSize,ISUNIT>(this_cache, blkSize, val);
         xshared[tid] = val;
      }
      else if(tid>=blkSize && n>blkSize && ii>=blkSize) {
         tocache_upr <T_ELEM, blkSize,threadsx*threadsy-blkSize, false, ISUNIT, CONJG>
            (threadsx*threadIdx.y+threadIdx.x-blkSize, &a[(ii-blkSize)*(lda+1)],
             lda, next_cache);
      }
      __syncthreads();
      /* Apply rectangular block (one thread per row) */
      if(n>blkSize && threadIdx.y==0 && threadIdx.x<ii) {
         T_ELEM val=0;
         for(int i=0; i<blkSize; i++) {
            val += rect[i*ii+threadIdx.x] * xshared[i];
         }
         xshared_actual[threadIdx.x] -= val;
      }
      __syncthreads();
      xshared-=blkSize;
      /* Switch caches for next iteration */
      if(this_cache==cache_even) {
         this_cache = cache_odd;
         next_cache = cache_even;
      } else {
         this_cache = cache_even;
         next_cache = cache_odd;
      }
   }
   /* Store x back to global memory */
   if(tid<n) xglobal[tid*incx] = xshared_actual[tid];
}

/*
 * Perform a triangular solve on an n x n matrix, where n%blkSize==0.
 */
template <typename T_ELEM, unsigned int n, unsigned int blkSize,
   unsigned int threadsx, unsigned int threadsy,
   bool ISUNIT, bool CONJG>
__launch_bounds__(threadsx*threadsy, 1)
void __global__ trsv_fixed_ut(const T_ELEM *a, int lda, T_ELEM *xglobal,
      int incx) {

   volatile T_ELEM __shared__ xshared_actual[n];
   volatile T_ELEM *xshared = xshared_actual;
   T_ELEM __shared__ cache_even[blkSize*blkSize];
   T_ELEM __shared__ cache_odd[blkSize*blkSize];
   T_ELEM __shared__ rect[(n>blkSize)? (n-blkSize)*blkSize : 1];
   int tid = threadsx*threadIdx.y+threadIdx.x;

   /* Precache x */
   if(tid<n) xshared[tid] = xglobal[tid*incx];

   /* Precache first block */
   tocache_upr <T_ELEM, blkSize,threadsx*threadsy, true, ISUNIT, CONJG>
      (threadsx*threadIdx.y+threadIdx.x, a, lda, cache_even);
   __syncthreads();

#pragma unroll
   for(int ii=0; ii<n; ii+=blkSize) {
      /* Preload entries for rectanular block */
      if(n>blkSize && threadIdx.y!=0)
         cacherect_trans <T_ELEM, blkSize, threadsx*(threadsy-1)>
            (threadsx*(threadIdx.y-1)+threadIdx.x, blkSize, n-ii-blkSize,
             a+blkSize*lda, lda, rect, n-blkSize);
      /* Solve diagonal block (one warp only) */
      if(tid<blkSize) {
         T_ELEM val = xshared[tid];
         if(ii%(2*blkSize)==0) {
            dblkSolve<T_ELEM,blkSize,ISUNIT>(cache_even, blkSize, val);
         } else {
            dblkSolve<T_ELEM,blkSize,ISUNIT>(cache_odd, blkSize, val);
         }
         xshared[tid] = val;
      }
      else if(n>blkSize && ii+blkSize<n) {
         if(ii%(2*blkSize)==0) {
            tocache_upr <T_ELEM,blkSize,threadsx*threadsy-32,true,ISUNIT,CONJG>
               (threadsx*threadIdx.y+threadIdx.x-32, &a[blkSize*(lda+1)], lda,
                cache_odd);
         } else {
            tocache_upr <T_ELEM,blkSize,threadsx*threadsy-32,true,ISUNIT,CONJG>
               (threadsx*threadIdx.y+threadIdx.x-32, &a[blkSize*(lda+1)], lda,
                cache_even);
         }
      }
      __syncthreads();
      /* Apply rectangular block (one thread per row) */
      if(n>blkSize && threadIdx.y==0 && ii+blkSize+threadIdx.x<n) {
         T_ELEM val=0;
         for(int i=0; i<blkSize; i++) {
            val += rect[(i)*(n-blkSize)+threadIdx.x] *
               xshared[i];
         }
         xshared[blkSize+threadIdx.x] -= val;
      }
      __syncthreads();
      a+=blkSize*(lda+1);
      xshared+=blkSize;
   }
   /* Store x back to global memory */
   if(tid<n) xglobal[tid*incx] = xshared_actual[tid];
}

/*
 * Perform a triangular solve on an n x n matrix.
 * Allows n%blkSize!=0 (unlike trsv_fixed_ln), but pays a performance hit
 * for doing so.
 */
template <typename T_ELEM, int maxn, int blkSize,
   int threadsx, int threadsy,
   bool ISUNIT, bool CONJG>
__launch_bounds__(threadsx*threadsy, 1)
void __global__ trsv_variable_ut(int n, const T_ELEM *a, int lda,
      T_ELEM *xglobal, int incx) {

   volatile T_ELEM __shared__ xshared_actual[maxn];
   volatile T_ELEM *xshared = xshared_actual;
   T_ELEM __shared__ cache_even[blkSize*blkSize];
   T_ELEM __shared__ cache_odd[blkSize*blkSize];
   T_ELEM __shared__ rect[(maxn>blkSize)? (maxn-blkSize)*blkSize : 1];
   int tid = threadsx*threadIdx.y+threadIdx.x;
   T_ELEM *this_cache, *next_cache;

   /* Precache x */
   if(tid<n) xshared[tid] = xglobal[tid*incx];

   /* Precache first block */
   this_cache = cache_even;
   next_cache = cache_odd;
   if(n<blkSize) {
      tocache_small_upr <T_ELEM, blkSize,threadsx*threadsy, true, ISUNIT, CONJG>
         (n, threadsx*threadIdx.y+threadIdx.x, a, lda, this_cache);
   } else {
      tocache_upr <T_ELEM, blkSize,threadsx*threadsy, true, ISUNIT, CONJG>
         (threadsx*threadIdx.y+threadIdx.x, a, lda, this_cache);
   }
   __syncthreads();

   for(int ii=0; ii<=n-blkSize; ii+=blkSize) {
      /* Preload entries for rectangular block */
      if(threadIdx.y!=0)
         cacherect_trans <T_ELEM, blkSize, threadsx*(threadsy-1)>
            (threadsx*(threadIdx.y-1)+threadIdx.x, blkSize, n-ii-blkSize,
             a+blkSize*lda, lda, rect, n-blkSize);
      /* Solve diagonal block (one warp only) */
      if(tid<blkSize) {
         T_ELEM val = xshared[tid];
         dblkSolve<T_ELEM,blkSize,ISUNIT>(this_cache, blkSize, val);
         xshared[tid] = val;
      }
      else if(ii+2*blkSize<=n)
         tocache_upr <T_ELEM, blkSize,threadsx*threadsy-32, true, ISUNIT, CONJG>
            (threadsx*threadIdx.y+threadIdx.x-32, &a[blkSize*(lda+1)], lda,
             next_cache);
      else if(ii+blkSize<n)
         tocache_small_upr <T_ELEM, blkSize,threadsx*threadsy-32, true, ISUNIT, CONJG>
            (n-(ii+blkSize), threadsx*threadIdx.y+threadIdx.x-32,
             &a[blkSize*(lda+1)], lda, next_cache);
      __syncthreads();
      /* Apply rectangular block (one thread per row) */
      if(threadIdx.y==0 && ii+blkSize+threadIdx.x<n) {
         T_ELEM val=0;
         for(int i=0; i<blkSize; i++) {
            val += rect[(i)*(n-blkSize)+threadIdx.x] *
               xshared[i];
         }
         xshared[blkSize+threadIdx.x] -= val;
      }
      __syncthreads();
      a+=blkSize*(lda+1);
      xshared+=blkSize;
      /* Swap caches */
      T_ELEM *temp = this_cache;
      this_cache = next_cache;
      next_cache = temp;
   }

   /* Final block */
   if(threadIdx.y==0 && threadIdx.x<blkSize) {
      volatile T_ELEM __shared__ xs;
      T_ELEM val = xshared[threadIdx.x];
      for(int i=0; i<n%blkSize; i++) {
         if(threadIdx.x==i) {
            if(!ISUNIT) val *= this_cache[i*blkSize+i];
            xs = val;
         }
         if(threadIdx.x>=i+1)
            val += this_cache[i*blkSize+threadIdx.x] * xs;
      }
      xshared[threadIdx.x] = val;
   }
   __syncthreads();

   /* Store x back to global memory */
   if(tid<n) xglobal[tid*incx] = xshared_actual[tid];
}

/* Performs trsv for Non-transposed Lower-triangular matrices
 * Requires trsv_init() to be called first to initialize sync[].
 */
template <typename T_ELEM, unsigned int nb, unsigned int threadsx,
   unsigned int threadsy, bool ISUNIT, bool CONJG>
__launch_bounds__(threadsx*threadsy, 4)
void __global__ trsv_ut_exec(int n, const T_ELEM *a, int lda,
      T_ELEM *xglobal, int incx, int *sync) {

   //int nblk = n / nb;
   int tid = threadsx*threadIdx.y + threadIdx.x;

   /* sync components:
    *    sync[0] => Last ready column [init to -1]
    *    sync[1] => Next row to assign [init to 0]
    */

   T_ELEM __shared__ partSum[threadsy*threadsx];
   T_ELEM __shared__ cache[nb*nb];
   volatile T_ELEM __shared__ soln[nb];
   T_ELEM __shared__ xlocal[nb];
   T_ELEM regcache[nb/threadsy];
   T_ELEM ps[nb/threadsy];

   if(incx<0) xglobal+=(1-n)*incx;

   /* Get row handled by this block */
   int row = nextRow(&sync[1]);

   bool short_row = ((n-1)/nb==row && n%nb!=0); /* requires special handling */

   /* Copy subdiagonal block into registers */
   if(row!=0) {
      const T_ELEM *aval = &a[(row*nb+threadIdx.x)*lda+(row-1)*nb+threadIdx.y];
#pragma unroll
      for(int j=0; j<nb; j+=threadsy)
         regcache[j/threadsy] = aval[j];
   }

   /* Copy diagonal block to shared memory */
   if(!short_row) {
      /* on a block row of full size */
      tocache_upr <T_ELEM,nb,threadsx*threadsy,true,ISUNIT,CONJG> (tid, &a[row*nb*lda+row*nb], lda, cache);
   } else {
      /* on last row, smaller than full blkSize */
      tocache_small_upr <T_ELEM,nb,threadsx*threadsy,true,ISUNIT,CONJG> (n%nb, tid, &a[row*nb*lda+row*nb], lda, cache);
   }
   __syncthreads();

#ifdef INV_AFTER
      if(row>=INV_AFTER) {
         T_ELEM __shared__ xsarray[128];
         invert<T_ELEM, nb, nb, threadsx, threadsy, ISUNIT, false>(cache, xsarray);
      }
#endif /* INV_AFTER */

   /* Loop over blocks as they become available */
   if(threadIdx.y==0) {
      if(!short_row) {
         soln[threadIdx.x] = xglobal[int(row*nb+threadIdx.x)*incx];
      } else {
         if(threadIdx.x<n%nb) soln[threadIdx.x] = xglobal[int(row*nb+threadIdx.x)*incx];
      }
   }
#pragma unroll
   for(int j=0; j<nb/threadsy; j++) ps[j] = 0;
   int col_done = -1;
   for(int col=0; col<row-1; col++) {
      /* apply update from block (row, col) */
      const T_ELEM *aval = &a[(row*nb+threadIdx.y)*lda + col*nb+threadIdx.x];
      T_ELEM *xg = &xglobal[int(col*nb)*incx];
      wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
      T_ELEM xl;
      xl = loadVolatile(&xg[int(threadIdx.x)*incx]);
      if(nb % threadsy == 0) {
        if(!short_row) {
#pragma unroll
            for(int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
               ps[j/threadsy] += aval[j*lda] * xl;
        } else {
            for(int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
               if(threadIdx.y+j<=n%nb) ps[j/threadsy] += aval[j*lda] * xl;
        }
      } else {
#pragma unroll
         for(int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
            if(j+threadIdx.y<nb)
               ps[j/threadsy] += aval[j*lda] * xl;
      }
   }
   T_ELEM val = 0;
#pragma unroll
   for(int i=0; i<nb; i+=threadsy) {
      partSum[threadIdx.x*threadsy+threadIdx.y] = ps[i/threadsy];
      __syncthreads();
      if(threadIdx.y==0 && threadIdx.x>=i && threadIdx.x<i+threadsy) {
         for(int j=0; j<nb; j++)
            val += partSum[(threadIdx.x-i)+threadsy*j];
      }
      __syncthreads();
   }
   if(row!=0) {
      const int col = row-1;
      /* apply update from block (row, col) */
      T_ELEM *xg = &xglobal[int(col*nb)*incx];
      wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
      T_ELEM *xl = xlocal+threadIdx.y;
      if(tid<nb) xlocal[tid] = loadVolatile(&xg[int(tid)*incx]);
      __syncthreads();
#pragma unroll
      for(int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
         val += regcache[j/threadsy] * xl[j];
   }
   partSum[threadIdx.y*threadsx+threadIdx.x] = val;
   __syncthreads();
   if(threadIdx.y==0) {
      for(int i=1; i<threadsy; i++)
         val += partSum[i*threadsx+threadIdx.x];
      val = soln[threadIdx.x]-val;
      if(short_row && threadIdx.x>=n%nb) val = 0.0;
   }

   /* Apply update from diagonal block (row, row) */
#ifdef INV_AFTER
      if(row>=INV_AFTER) {
         slvinv<T_ELEM, nb, threadsy>(cache, xlocal, val, partSum);
         if(!short_row || threadIdx.x<n%nb) {
            if(threadIdx.y==0) {
               xglobal[int(row*nb+tid)*incx] = val;
            }
         }
      } else {
         if(threadIdx.y==0) {
            dblkSolve<T_ELEM,nb,ISUNIT>(cache, nb, val);
            if(!short_row || threadIdx.x<n%nb) {
               xglobal[int(row*nb+tid)*incx] = val;
            }
         }
      }
#else /* INV_AFTER */
      if(threadIdx.y==0) {
         dblkSolve<T_ELEM,nb,ISUNIT>(cache, nb, val);
         if(!short_row || threadIdx.x<n%nb) {
            xglobal[int(row*nb+tid)*incx] = val;
         }
      }
#endif /* INV_AFTER */
   /* Notify other blocks that soln is ready for this row */
   __threadfence(); // Wait for xglobal to be visible to other blocks
   if(tid==0) atomicAdd(&sync[0],1); // Use atomicAdd to bypass L1 miss
   __threadfence(); // Flush sync[0] asap

}

/* Performs trsv for Transposed Lower-triangular matrices
 * Requires trsv_init() to be called first to initialize sync[].
 */
template <typename T_ELEM, unsigned int nb, unsigned int threadsx,
   unsigned int threadsy, bool ISUNIT, bool CONJG>
__launch_bounds__(threadsx*threadsy, 4)
void __global__ trsv_un_exec(int n, const T_ELEM *a, int lda,
      T_ELEM *xglobal, int incx, int *sync) {

   int nblk = (n-1) / nb + 1;
   int tid = threadsx*threadIdx.y + threadIdx.x;

   /* sync components:
    *    sync[0] => nblk - Last ready column [init to -1]
    *    sync[1] => nblk - Next row to assign [init to 0]
    */

   T_ELEM __shared__ partSum[threadsy*threadsx];
   T_ELEM __shared__ cache[nb*nb];
   T_ELEM __shared__ xlocal[nb];
   T_ELEM regcache[nb/threadsy];
   if(incx<0) xglobal+=(n-1)*(-incx); // if incx negative, start at end

   /* Get row handled by this block */
   int row = nblk-1 - nextRow(&sync[1]);

   bool short_row = ((n-1)/nb==row && n%nb!=0); /* requires special handling */

   /* Copy subdiagonal block into registers */
   if(row!=nblk-1) {
      const T_ELEM *aval = &a[((row+1)*nb+threadIdx.y)*lda+row*nb+threadIdx.x];
#pragma unroll
      for(int j=0; j<nb; j+=threadsy)
         regcache[j/threadsy] = aval[j*lda];
   }

   /* Copy diagonal block to shared memory */
   if(!short_row) {
      /* on a block row of full size */
#ifdef INV_AFTER
      if(nblk-1-row>=INV_AFTER) {
         tocache_upr <T_ELEM,nb,threadsx*threadsy,true,ISUNIT,CONJG> (tid, &a[row*nb*lda+row*nb], lda, cache);
      } else {
         tocache_upr <T_ELEM,nb,threadsx*threadsy,false,ISUNIT,CONJG> (tid, &a[row*nb*lda+row*nb], lda, cache);
      }
#else /* INV_AFTER */
      tocache_upr <T_ELEM,nb,threadsx*threadsy,false,ISUNIT,CONJG> (tid, &a[row*nb*lda+row*nb], lda, cache);
#endif /* INV_AFTER */
   } else {
      /* on last row, smaller than full blkSize */
#ifdef INV_AFTER
      if(nblk-1-row>=INV_AFTER) {
         tocache_small_upr <T_ELEM,nb,threadsx*threadsy,true,ISUNIT,CONJG> (n%nb, tid, &a[row*nb*lda+row*nb], lda, cache);
      } else {
         tocache_small_upr <T_ELEM,nb,threadsx*threadsy,false,ISUNIT,CONJG> (n%nb, tid, &a[row*nb*lda+row*nb], lda, cache);
      }
#else /* INV_AFTER */
      tocache_small_upr <T_ELEM,nb,threadsx*threadsy,false,ISUNIT,CONJG> (n%nb, tid, &a[row*nb*lda+row*nb], lda, cache);
#endif /* INV_AFTER */
   }
   __syncthreads();

#ifdef INV_AFTER
   if(nblk-1-row>=INV_AFTER) {
      T_ELEM __shared__ xsarray[128];
      invert<T_ELEM, nb, nb, threadsx, threadsy, ISUNIT, true>(cache, xsarray);
   }
#endif /* INV_AFTER */

   /* Loop over blocks as they become available */
   T_ELEM val = 0;
   if(threadIdx.y==0) {
      if(!short_row) {
         val = -xglobal[int(row*nb+threadIdx.x)*incx];
      } else {
         if(threadIdx.x<n%nb)
            val = -xglobal[int(row*nb+threadIdx.x)*incx];
         else 
            val = 0;
      }
   }
   int col_done = -1;
   for(int col=nblk-1; col>row+1; col--) {
      /* apply update from block (row, col) */
      const T_ELEM *aval = &a[(col*nb+threadIdx.y)*lda + row*nb+threadIdx.x];
      T_ELEM *xg = &xglobal[int(col*nb)*incx];
      wait_until_ge(tid, &sync[0], nblk-1-col, &col_done); // Wait for diagonal block to be done
      T_ELEM *xl = xlocal+threadIdx.y;
      if(col<nblk-1) {
         xlocal[tid] = loadVolatile(&xg[int(tid)*incx]);
      } else {
         if(threadIdx.x<(n-1)%nb+1) xlocal[tid] = loadVolatile(&xg[int(tid)*incx]);
         else                       xlocal[tid] = 0;
      }
      __syncthreads();
      if(nb % threadsy == 0) {
         if(col==nblk-1) {
#pragma unroll
            for(int j=0; j<nb; j+=threadsy) {
               if(col*nb+threadIdx.y+j>=n) continue; // avoid going off end of array
               val += aval[j*lda] * xl[j];
            }
         } else {
#pragma unroll
            for(int j=0; j<nb; j+=threadsy) {
               val += aval[j*lda] * xl[j];
            }
         }
      } else {
#pragma unroll
         for(int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
            if(j+threadIdx.y<nb) val += aval[j*lda] * xl[j];
      }
   }
   if(row!=nblk-1) {
      const int col = row+1;
      /* apply update from block (row, col) */
      T_ELEM *xg = &xglobal[int(col*nb)*incx];
      wait_until_ge(tid, &sync[0], nblk-1-col, &col_done); // Wait for diagonal block to be done
      T_ELEM *xl = xlocal+threadIdx.y;
      if(col<nblk-1) {
         if(tid<nb) xlocal[tid] = loadVolatile(&xg[int(tid)*incx]);
         __syncthreads();
#pragma unroll
         for(int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
            val += regcache[j/threadsy] * xl[j];
      } else {
         if(tid<(n-1)%nb+1) xlocal[tid] = loadVolatile(&xg[int(tid)*incx]);
         __syncthreads();
#pragma unroll
         for(int j=0; j<(n-1)%nb+1; j+=threadsy) // do j=0,nb-1,threadsy
            if(j+threadIdx.y<(n-1)%nb+1) val += regcache[j/threadsy] * xl[j];
      }
   }
   partSum[threadIdx.y*threadsx+threadIdx.x] = val;
   __syncthreads();
   if(threadIdx.y==0) {
      for(int i=1; i<threadsy; i++)
         val += partSum[i*threadsx+threadIdx.x];
      val = -val;
   }

   /* Apply update from diagonal block (row, row) */
#ifdef INV_AFTER
      if(nblk-1-row>=INV_AFTER) {
         slvinv_trans<T_ELEM, nb, threadsy>(cache, xlocal, val, partSum, row);
         if(!short_row || threadIdx.x<n%nb) {
            if(threadIdx.y==0) {
               xglobal[int(row*nb+tid)*incx] = val;
            }
         }
      } else {
         if(threadIdx.y==0) {
            dblkSolve_trans<T_ELEM,nb,ISUNIT>(cache, nb, val);
            if(!short_row || threadIdx.x<n%nb) {
               xglobal[int(row*nb+tid)*incx] = val;
            }
         }
      }
#else /* INV_AFTER */
      if(threadIdx.y==0) {
         dblkSolve_trans<T_ELEM,nb,ISUNIT>(cache, nb, val);
         if(!short_row || threadIdx.x<n%nb) {
            xglobal[int(row*nb+tid)*incx] = val;
         }
      }
#endif /* INV_AFTER */
   /* Notify other blocks that soln is ready for this row */
   __threadfence(); // Wait for xglobal to be visible to other blocks
   if(tid==0) atomicAdd(&sync[0],1); // Use atomicAdd to bypass L1 miss
   __threadfence(); // Flush sync[0] asap
}

#ifdef TIMING
void output_trace(int nblk, unsigned int prof[]);
#endif

void ASEArch_dtrsv(enum ASEArch_uplo uplo, enum ASEArch_trans trans,
      enum ASEArch_diag diag, int n, const double a[], int lda, double x[],
      int incx) {

   if(n<SWITCH_TO_GMEM) {
      if(incx < 0) x += (n-1)*(-incx); // if incx is negative, start at end
      dim3 nblocks(1);
      dim3 nthreads(THREADSX,THREADSY);
      if(uplo==ASEARCH_LWR) {
         if(trans==ASEARCH_NONTRANS) {
            for(int i=0; i<n; i+=NB) {
               if(diag==ASEARCH_UNIT) { /* Lwr, NonTrans, Unit */
                  if(NB<=n-i) {
                     trsv_fixed_ln <double,NB,NBI,THREADSX,THREADSY,true,false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_ln <double,NB,NBI,THREADSX,THREADSY,true,false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               } else { /* Lwr, NonTrans, Non-unit */
                  if(NB<=n-i) {
                     trsv_fixed_ln <double, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_ln <double, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               }

               if(i+NB<n) {
                  const double *a2 = &a[i*(lda+1)+NB];
                  const double *x2 = (incx>0) ? &x[i*incx] : &x[(i+MIN(NB,n-i)-1)*incx];
                  double *y2 = (incx>0) ? &x[(i+NB)*incx] : &x[(n-1)*incx];
                  ASEArch_dgemv(ASEARCH_NONTRANS, n-i-NB, MIN(NB,n-i), -1.0, a2,
                        lda, x2, incx, 1.0, y2, incx);
               }
            }
         } else { /* Lwr, Trans */
            for(int i=((n-1)/NB)*NB; i>=0; i-=NB) {
               if(diag==ASEARCH_UNIT) { /* Lwr, Trans, Unit */
                  if(NB<=n-i) {
                     trsv_fixed_lt <double, NB, NBI, THREADSX, THREADSY, true, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_lt <double, NB, NBI, THREADSX, THREADSY, true, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               } else { /* Lwr, Trans, Non-unit */
                  if(NB<=n-i) {
                     trsv_fixed_lt <double, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_lt <double, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               }

               const double *a2 = &a[i];
               const double *x2 = (incx>0) ? &x[i*incx] : &x[(i+MIN(NB,n-i)-1)*incx];
               double *y2 = (incx>0) ? x : &x[(i-1)*incx];
               if(i>0) {
                  ASEArch_dgemv(ASEARCH_TRANS, MIN(NB,n-i), i, -1.0, a2, lda,
                        x2, incx, 1.0, y2, incx);
               }
            }
         }
      } else { /* Upr */
         if(trans==ASEARCH_NONTRANS) { /* Upr, NonTrans */
            for(int i=((n-1)/NB)*NB; i>=0; i-=NB) {
               if(diag==ASEARCH_UNIT) { /* Upr, NonTrans, Unit */
                  if(NB<=n-i) {
                     trsv_fixed_un <double, NB, NBI, THREADSX, THREADSY, true, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_un <double, NB, NBI, THREADSX, THREADSY, true, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               } else { /* Upr, NonTrans, Non-unit */
                  if(NB<=n-i) {
                     trsv_fixed_un <double, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_un <double, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               }

               const double *a2 = &a[i*lda];
               const double *x2 = (incx>0) ? &x[i*incx] : &x[(i+MIN(NB,n-i)-1)*incx];
               double *y2 = (incx>0) ? x : &x[(i-1)*incx];
               if(i>0) {
                  ASEArch_dgemv(ASEARCH_NONTRANS, i, MIN(NB,n-i), -1.0, a2, lda,
                        x2, incx, 1.0, y2, incx);
               }
            }
         } else { /* Upr, Trans */
            for(int i=0; i<n; i+=NB) {
               if(diag==ASEARCH_UNIT) { /* Upr, Trans, Unit */
                  if(NB<=n-i) {
                     trsv_fixed_ut <double,NB,NBI,THREADSX,THREADSY,true,false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_ut <double,NB,NBI,THREADSX,THREADSY,true,false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               } else { /* Upr, Trans, Non-unit */
                  if(NB<=n-i) {
                     trsv_fixed_ut <double, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_ut <double, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               }

               if(i+NB<n) {
                  const double *a2 = &a[i*(lda+1)+NB*lda];
                  const double *x2 = (incx>0) ? &x[i*incx] : &x[(i+MIN(NB,n-i)-1)*incx];
                  double *y2 = (incx>0) ? &x[(i+NB)*incx] : &x[(n-1)*incx];
                  ASEArch_dgemv(ASEARCH_TRANS, MIN(NB,n-i), n-i-NB, -1.0, a2,
                        lda, x2, incx, 1.0, y2, incx);
               }
            }
         }
      }
   } else { /* Big enough to use _exec call */
      int *sync;
      int nblk = (n-1)/NB_TASK+1;
      CudaSafeCall( cudaMalloc(&sync, 2*sizeof(int)) );
      trsv_init <<<1,1>>> (sync);
      if(uplo==ASEARCH_LWR) {
         if(trans==ASEARCH_NONTRANS) { /* Lwr, Non-Transpose */
#ifdef TIMING
            unsigned int *prof_gpu=NULL;
            CudaSafeCall( cudaMalloc(&prof_gpu, (3*nblk*nblk+2*nblk)*sizeof(unsigned int)) );
#endif /* TIMING */
            if(diag==ASEARCH_UNIT) { /* Lwr, NonTrans, Unit */
               trsv_ln_exec <double,NB_TASK,THREADSX_TASK,THREADSY_TASK,true,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync
#ifdef TIMING
                     , prof_gpu
#endif
                     );
            } else { /* Lwr, NonTrans, Non-Unit */
               trsv_ln_exec <double,NB_TASK,THREADSX_TASK,THREADSY_TASK,false,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync
#ifdef TIMING
                     , prof_gpu
#endif
                     );
            }
#ifdef TIMING
            unsigned int *prof = (unsigned int *) malloc((3*nblk*nblk+2*nblk)*sizeof(unsigned int));
            CudaSafeCall( 
                  cudaMemcpy(prof, prof_gpu, (3*nblk*nblk+2*nblk)*sizeof(unsigned int),
                     cudaMemcpyDeviceToHost)
                  );
            CudaSafeCall( cudaFree(prof_gpu) );
            output_trace(nblk, prof);
            //output_trace2(nblk, prof);
            free(prof);
#endif
         } else { /* Lwr, Transpose */
            if(diag==ASEARCH_UNIT) { /* Lwr, NonTrans, Unit */
               trsv_lt_exec <double,NB_TASK,THREADSX_TASK,THREADSY_TASK,true,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync);
            } else { /* Lwr, NonTrans, Non-Unit */
               trsv_lt_exec <double,NB_TASK,THREADSX_TASK,THREADSY_TASK,false,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync);
            }
         }
      } else { /* Upr */
         if(trans==ASEARCH_NONTRANS) { /* Upr, Non-Transpose */
            if(diag==ASEARCH_UNIT) { /* Lwr, NonTrans, Unit */
               trsv_un_exec <double,NB_TASK,THREADSX_TASK,THREADSY_TASK,true,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync);
            } else { /* Upr, NonTrans, Non-Unit */
               trsv_un_exec <double,NB_TASK,THREADSX_TASK,THREADSY_TASK,false,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync);
            }
         } else { /* Upr, Transpose */
            if(diag==ASEARCH_UNIT) { /* Upr, NonTrans, Unit */
               trsv_ut_exec <double,NB_TASK,THREADSX_TASK,THREADSY_TASK,true,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync);
            } else { /* Upr, NonTrans, Non-Unit */
               trsv_ut_exec <double,NB_TASK,THREADSX_TASK,THREADSY_TASK,false,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync);
            }
         }
      }
      CudaSafeCall( cudaFree(sync) );
   }
}

void ASEArch_strsv(enum ASEArch_uplo uplo, enum ASEArch_trans trans,
      enum ASEArch_diag diag, int n, const float a[], int lda, float x[],
      int incx) {

   if(n<SWITCH_TO_GMEM) {
      if(incx < 0) x += (n-1)*(-incx); // if incx is negative, start at end
      dim3 nblocks(1);
      dim3 nthreads(THREADSX,THREADSY);
      if(uplo==ASEARCH_LWR) {
         if(trans==ASEARCH_NONTRANS) {
            for(int i=0; i<n; i+=NB) {
               if(diag==ASEARCH_UNIT) { /* Lwr, NonTrans, Unit */
                  if(NB<=n-i) {
                     trsv_fixed_ln <float,NB,NBI,THREADSX,THREADSY,true,false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_ln <float,NB,NBI,THREADSX,THREADSY,true,false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               } else { /* Lwr, NonTrans, Non-unit */
                  if(NB<=n-i) {
                     trsv_fixed_ln <float, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_ln <float, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               }

               if(i+NB<n) {
                  const float *a2 = &a[i*(lda+1)+NB];
                  const float *x2 = (incx>0) ? &x[i*incx] : &x[(i+MIN(NB,n-i)-1)*incx];
                  float *y2 = (incx>0) ? &x[(i+NB)*incx] : &x[(n-1)*incx];
                  ASEArch_sgemv(ASEARCH_NONTRANS, n-i-NB, MIN(NB,n-i), -1.0, a2,
                        lda, x2, incx, 1.0, y2, incx);
               }
            }
         } else { /* Lwr, Trans */
            for(int i=((n-1)/NB)*NB; i>=0; i-=NB) {
               if(diag==ASEARCH_UNIT) { /* Lwr, Trans, Unit */
                  if(NB<=n-i) {
                     trsv_fixed_lt <float, NB, NBI, THREADSX, THREADSY, true, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_lt <float, NB, NBI, THREADSX, THREADSY, true, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               } else { /* Lwr, Trans, Non-unit */
                  if(NB<=n-i) {
                     trsv_fixed_lt <float, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_lt <float, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               }

               const float *a2 = &a[i];
               const float *x2 = (incx>0) ? &x[i*incx] : &x[(i+MIN(NB,n-i)-1)*incx];
               float *y2 = (incx>0) ? x : &x[(i-1)*incx];
               if(i>0) {
                  ASEArch_sgemv(ASEARCH_TRANS, MIN(NB,n-i), i, -1.0, a2, lda,
                        x2, incx, 1.0, y2, incx);
               }
            }
         }
      } else { /* Upr */
         if(trans==ASEARCH_NONTRANS) { /* Upr, NonTrans */
            for(int i=((n-1)/NB)*NB; i>=0; i-=NB) {
               if(diag==ASEARCH_UNIT) { /* Upr, NonTrans, Unit */
                  if(NB<=n-i) {
                     trsv_fixed_un <float, NB, NBI, THREADSX, THREADSY, true, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_un <float, NB, NBI, THREADSX, THREADSY, true, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               } else { /* Upr, NonTrans, Non-unit */
                  if(NB<=n-i) {
                     trsv_fixed_un <float, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_un <float, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               }

               const float *a2 = &a[i*lda];
               const float *x2 = (incx>0) ? &x[i*incx] : &x[(i+MIN(NB,n-i)-1)*incx];
               float *y2 = (incx>0) ? x : &x[(i-1)*incx];
               if(i>0) {
                  ASEArch_sgemv(ASEARCH_NONTRANS, i, MIN(NB,n-i), -1.0, a2, lda,
                        x2, incx, 1.0, y2, incx);
               }
            }
         } else { /* Upr, Trans */
            for(int i=0; i<n; i+=NB) {
               if(diag==ASEARCH_UNIT) { /* Upr, Trans, Unit */
                  if(NB<=n-i) {
                     trsv_fixed_ut <float,NB,NBI,THREADSX,THREADSY,true,false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_ut <float,NB,NBI,THREADSX,THREADSY,true,false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               } else { /* Upr, Trans, Non-unit */
                  if(NB<=n-i) {
                     trsv_fixed_ut <float, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (&a[i*(lda+1)], lda, &x[i*incx], incx);
                  } else {
                     trsv_variable_ut <float, NB, NBI, THREADSX, THREADSY, false, false>
                        <<<1,dim3(THREADSX, THREADSY)>>>
                        (n-i, &a[i*(lda+1)], lda, &x[i*incx], incx);
                  }
               }

               if(i+NB<n) {
                  const float *a2 = &a[i*(lda+1)+NB*lda];
                  const float *x2 = (incx>0) ? &x[i*incx] : &x[(i+MIN(NB,n-i)-1)*incx];
                  float *y2 = (incx>0) ? &x[(i+NB)*incx] : &x[(n-1)*incx];
                  ASEArch_sgemv(ASEARCH_TRANS, MIN(NB,n-i), n-i-NB, -1.0, a2,
                        lda, x2, incx, 1.0, y2, incx);
               }
            }
         }
      }
   } else { /* Big enough to use _exec call */
      int *sync;
      int nblk = (n-1)/NB_TASK+1;
      CudaSafeCall( cudaMalloc(&sync, 2*sizeof(int)) );
      trsv_init <<<1,1>>> (sync);
      if(uplo==ASEARCH_LWR) {
         if(trans==ASEARCH_NONTRANS) { /* Lwr, Non-Transpose */
#ifdef TIMING
            unsigned int *prof_gpu=NULL;
            CudaSafeCall( cudaMalloc(&prof_gpu, (3*nblk*nblk+2*nblk)*sizeof(unsigned int)) );
#endif /* TIMING */
            if(diag==ASEARCH_UNIT) { /* Lwr, NonTrans, Unit */
               trsv_ln_exec <float,NB_TASK,THREADSX_TASK,THREADSY_TASK,true,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync
#ifdef TIMING
                     , prof_gpu
#endif
                     );
            } else { /* Lwr, NonTrans, Non-Unit */
               trsv_ln_exec <float,NB_TASK,THREADSX_TASK,THREADSY_TASK,false,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync
#ifdef TIMING
                     , prof_gpu
#endif
                     );
            }
#ifdef TIMING
            unsigned int *prof = (unsigned int *) malloc((3*nblk*nblk+2*nblk)*sizeof(unsigned int));
            CudaSafeCall( 
                  cudaMemcpy(prof, prof_gpu, (3*nblk*nblk+2*nblk)*sizeof(unsigned int),
                     cudaMemcpyDeviceToHost)
                  );
            CudaSafeCall( cudaFree(prof_gpu) );
            output_trace(nblk, prof);
            //output_trace2(nblk, prof);
            free(prof);
#endif
         } else { /* Lwr, Transpose */
            if(diag==ASEARCH_UNIT) { /* Lwr, NonTrans, Unit */
               trsv_lt_exec <float,NB_TASK,THREADSX_TASK,THREADSY_TASK,true,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync);
            } else { /* Lwr, NonTrans, Non-Unit */
               trsv_lt_exec <float,NB_TASK,THREADSX_TASK,THREADSY_TASK,false,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync);
            }
         }
      } else { /* Upr */
         if(trans==ASEARCH_NONTRANS) { /* Upr, Non-Transpose */
            if(diag==ASEARCH_UNIT) { /* Lwr, NonTrans, Unit */
               trsv_un_exec <float,NB_TASK,THREADSX_TASK,THREADSY_TASK,true,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync);
            } else { /* Upr, NonTrans, Non-Unit */
               trsv_un_exec <float,NB_TASK,THREADSX_TASK,THREADSY_TASK,false,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync);
            }
         } else { /* Upr, Transpose */
            if(diag==ASEARCH_UNIT) { /* Upr, NonTrans, Unit */
               trsv_ut_exec <float,NB_TASK,THREADSX_TASK,THREADSY_TASK,true,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync);
            } else { /* Upr, NonTrans, Non-Unit */
               trsv_ut_exec <float,NB_TASK,THREADSX_TASK,THREADSY_TASK,false,false> <<<nblk, dim3(THREADSX_TASK,THREADSY_TASK)>>> (n, a, lda, x, incx, sync);
            }
         }
      }
      CudaSafeCall( cudaFree(sync) );
   }
}

#ifdef TIMING
void output_trace(int nblk, unsigned int prof[]) {
   const int NSM = 14;

   int *tinv = (int *) malloc(nblk*sizeof(int));
   unsigned int *smmap = &prof[2*(nblk+1)*(nblk+1)];

   /* prof[2*(nblk+1)*(nblk+1) + row]  SM of block row
    * prof[2*((nblk+1)*row + col)]     point to length two sa, en array.
    *                                  init is treated as col=nblk.
    */

   unsigned int smsa[NSM];
   for(int i=0; i<NSM; i++) smsa[i] = UINT_MAX;
   for(int row=0; row<nblk; row++) {
      int sm = smmap[row]; // SM for row
      unsigned int *p = &prof[2*(nblk+1)*row];
      if(p[2*nblk] < smsa[sm]) smsa[sm] = p[2*nblk];
   }

   for(int row=0; row<nblk; row++) {
      int sm = smmap[row]; // SM for row
      unsigned int *p = &prof[2*(nblk+1)*row];
      for(int col=0; col<row; col++) {
         printf("%4u%12u%12u RT %12d%12d\n", sm, p[2*col+0]-smsa[sm],
               p[2*col+1]-smsa[sm], row, col);
      }
      printf("%4u%12u%12u DI %12d%12d\n", sm, p[2*row+0]-smsa[sm],
            p[2*row+1]-smsa[sm], row, row);
      printf("%4u%12d%12d CI %12d\n", sm, p[2*nblk+0]-smsa[sm],
            p[2*nblk+1]-smsa[sm], row);
   }

}
#endif
