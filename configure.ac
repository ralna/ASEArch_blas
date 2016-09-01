AC_INIT(ASEArch_blas, 1.0.0, jonathan.hogg@stfc.ac.uk)
AC_CONFIG_SRCDIR([src/dtrsv.cu])
AM_INIT_AUTOMAKE(ASEArch_blas, 1.0.0)

# Check for generic compilers
AC_PROG_CC
AC_PROG_CXX
AC_PROG_RANLIB

# Check for CUDA compiler
AC_CHECK_PROGS(NVCC,nvcc)

# Check for external libraries (e.g. BLAS, LAPACK, MeTiS)
ACX_BLAS(,[AC_MSG_WARN([No BLAS library found. make check will not work.])])
ACX_LAPACK(,[AC_MSG_WARN([No LAPACK library found. make check will not work.])])

# Output data
AC_CONFIG_FILES(Makefile)
AC_OUTPUT
