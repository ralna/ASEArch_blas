## ASEArch: The CCP in Advanced Software for Emerging Architectures

ASEArch is a Collabrative Computational Project funded by the UK's
Engineering and Physical Sciences Research Council (EPSRC).
This work was carried out under grant EP/J010553/1.

# ASEArch BLAS
Our initial release provides high-speed versions of triangular solve (_TRSV)
that significantly outperforms the current version of CUBLAS.

In addition source code for matrix-vector multiplication (_GEMV) is provided,
however the performance is slightly lower than the CUBLAS, which should be
used in preference.

Both FERMI and KEPLER architectures are supported, however performance on
GK104 based Kepler cards is appaling due to high cost of __threadfence().
GK110 performance is apparently better, but we haven't tested it yet.
There are no plans to add support for the older architectures.

## COMPILATION
The library may be compiled and installed as described in the accompanying
INSTALL file. In summary, the command `make` will compile the file
libasearch_blas.a which can then be linked in to your program.

The command `make test` will run a test to ensure correct answers are obtained.

## USAGE
See the doc directory for a description of the API, which is very simlar to
that of the BLAS.
