EXTRA_DIST = doc/blas.tex doc/blas.pdf example/bench_dtrsv.cu

lib_LIBRARIES = libASEArch_blas.a

libASEArch_blas_a_SOURCES = \
	src/dtrsv.cu \
	src/dgemv.cu \
	include/ASEArch_blas.h \
	include/safecall.h

TESTS = \
	dtrsv_full_test

check_PROGRAMS = \
	dtrsv_full_test

dtrsv_full_test_SOURCES = src/test/dtrsv_full_test.cu include/ASEArch_blas.h
dtrsv_full_test_LDADD = -L. -lASEArch_blas $(LAPACK_LIBS) $(BLAS_LIBS)
dtrsv_full_test_LINK = $(NVCC) $(NVCC_FLAGS) $(AM_NVCC_FLAGS) $(AM_LD_FLAGS) $(LDFLAGS) -o $@

AM_NVCC_FLAGS = -Iinclude -arch=sm_20
AM_OBJ_FLAGS = -dc
AM_LD_FLAGS = -lcuda

.cu.o:
	$(NVCC) $(NVCC_FLAGS) $(AM_NVCC_FLAGS) $(AM_OBJ_FLAGS) -o $@ $<
