#ifndef BLAS2_H
#define BLAS2_H

enum ASEArch_trans {
   ASEARCH_TRANS,
   ASEARCH_NONTRANS,
};

enum ASEArch_uplo {
   ASEARCH_UPR,
   ASEARCH_LWR,
};

enum ASEArch_diag {
   ASEARCH_UNIT,
   ASEARCH_NONUNIT,
};

void ASEArch_dgemv(enum ASEArch_trans trans, int m, int n, double alpha,
      const double a[], int lda, const double x[], int incx, double beta,
      double y[], int incy);
void ASEArch_dtrsv(enum ASEArch_uplo uplo, enum ASEArch_trans trans,
      enum ASEArch_diag diag, int n, const double a[], int lda, double x[],
      int incx);

void ASEArch_sgemv(enum ASEArch_trans trans, int m, int n, float alpha,
      const float a[], int lda, const float x[], int incx, float beta,
      float y[], int incy);
void ASEArch_strsv(enum ASEArch_uplo uplo, enum ASEArch_trans trans,
      enum ASEArch_diag diag, int n, const float a[], int lda, float x[],
      int incx);

#endif
