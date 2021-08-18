# Introduction to Scientific Programming

https://en.wikipedia.org/wiki/Scientific_programming_language

Loose def: a scientific programming language is designed and optimized for the use of mathematical formula and matrices.[2] 

Scientific programming languages in the stronger sense include ALGOL, APL, Fortran, J, Julia, Maple, MATLAB and R.

Nowadays, the concept of matrices and linear algebra is extended to a concept of efficient data storage and efficient operations over them.

Key requirements:
- fast execution of the code (complex algorithms)
- ease of code reuse / restructuring 
- reproducibility of the results

Contrast to general-purpose language:
- less concern with public/private separation
- less concern with ABI 

## TODO
Zero cost abstraction - Rackcaucas


### Example
In many applications, we encounter the task of optimization a function given by a routine (e.g. engineering, finance, etc.)

```
using Optim

P(x,y) = x^2 - 3x*y + 5y^2 - 7y + 3   # user defined function

z₀ = [ 0.0
       0.0 ]     # starting point for optimization algorithm

optimize(z -> P(z...), z₀, Newton())
#optimize(z -> P(z...), z₀, Newton();autodiff = :forward)
#optimize(z -> P(z...), z₀, ConjugateGradient())

```

Very simple for a user, very complicated for a programmer. The program should:
 - compute gradient (Hessian) of a user function
 - pick the right optimization method

Classical thinking: create a library, call it.

Think of an experiment: ```main``` taking a configuration file. The configuration file can be simple: ```input file```, what to do with it, ```output file```.

The more complicated experiments you want to do, the more complex your configuration file becomes. Sooner or later, you will create a new *configuration language*, or *scripting language*.

Ending up in *2 language problem*. 


1.  Low-level programming = computer centric
    - close to the hardware
    - allows excellent optimization for fast execution

2. High-level programming = user centric
    - experimenting = running multiple configurations
    - running code with many different parameters as easily as possible

In scientific programming the most well known scipting languages are: Python,  Matlab, R

- If you care about standard "configurations" they are just perfect. 
- You hit a problem with more complex experiments. 

The scripting language typically makes decisions (```if```) at runtime. Becomes slow.

![](julia-scope.pdf)

## Other approaches
1. Just in time compilation (HL -> LL)
2. automatic typing (auto in C++) (LL->HL)



## Example integer division

Indexing array x:
```
y=x[4/2]
```


![](processor.gif)

cache misses.


Other reason: no threads, poor use of SSE...

### Can be done much better

Different algorithms: 
https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
Architecture sepcific optimizations:
![](bench_incremental.svg)

Microkenels from oepnblas:
dgemm_kernel_16x2_haswell.S     
dgemm_kernel_4x4_haswell.S  
dgemm_kernel_4x8_haswell.S  
dgemm_kernel_4x8_sandy.S    
dgemm_kernel_6x4_piledriver.S   
dgemm_kernel_8x2_bulldozer.S    
dgemm_kernel_8x2_piledriver.S

### Blas specializes matrix multiplication routines

Depending on shape:
gemm | Computes a matrix-matrix product with general matrices.
symm | Computes a matrix-matrix product where one input matrix is symmetric and one matrix is general.

trmm | Computes a matrix-matrix product where one input matrix is triangular and one input matrix is general.

Depending on type:
Short-precision real	SGEMM
Long-precision real  	DGEMM
Short-precision complex	CGEMM
Long-precision complex	ZGEMM

Full syntax:
```
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);
```

## Quest for generality

BLAS is fast. Is it general?

Consider implementation of a quadratic form:
$$
Q = A*P*A'
$$
 You should call:
 -  ```gemm``` for general matrices.
 -  ```symm``` for symmetric matrices.
 -  ```trmm``` for triangular matrices.


 If you know what your types are you can write it by hand. General code in C is hard.

 Python to the rescue:
 Wrapper object ndarry 




### 2 language problem

Writing very efficient code requires to go down to hardware specifics -> low-level language.
Efficient code restructuring is possible when the language allows easy to handle  abstraction -> high-level language.

Classical solution = 2 languages (Matlab/C, Python/C,C++, R/C)

### BLAS
This phenomenon was recognized as early as in 1979 in linear algebra. Original Fortran library for "Basic Linear Algebra Subprograms" became standardized since then and became de-facto standard for high-performance computing.

Defines 3 levels of algebra:
1. vector operations (addition, axpy=> y=a*x+y, etc.)
2. vector/matrix operations (multiplication, gemv=>y=aAx+by)
3. matrix/matrix operations (gemm=>)

![](blas_benchmark.png)

Used in Matlab, R, NumPy.

Why si BLAS so fast??
- special algorithms optimized for SSE (single instruction multiple data) instructions
    + 16 resisters of SSE imply blocking matrices into   4 blocks in parallel.
- memory alignment (cache utilization)

http://www.mathematik.uni-ulm.de/~lehn/apfel/sghpc/gemm/


# Advantages and disadvantages
 1. compilation
    + very fast code
    - slow interaction (caching...)
    - libraries are harder
    - debugging will be harder
    -  ![](julia-compilation.png)


 2. Multiple dispatch
    + allows great extensibility and code composition
    - not (yet) mainstream thinking


# Syntax

Syntactic Sugar:
Cheat sheet: https://cheatsheets.quantecon.org/

# Typing -> Lecture 2

- static
- dynamic

# 

# Packages
Alternative to libraries in C, metadata

### tough lab session
