<p align="center">
 <img src="https://raw.githubusercontent.com/JuliaTeachingCTU/JuliaCTUGraphics/main/logo/Scientific-Programming-in-Julia-logo.svg" alt="Course logo"/>
</p>

---

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/master/LICENSE)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaTeachingCTU.github.io/Scientific-Programming-in-Julia/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaTeachingCTU.github.io/Scientific-Programming-in-Julia/dev)


This repository will contain all the course materials for the Julia course taught at FEL.
Please go ahead and add your thoughts.

## Pevnak

The goal of the course should be to learn students to think in Julia and teach them design patterns.

## Syllabus

### Introduction (Recap of Bachelor course lecture 1-6)
  - Syntactic sugar
  - Function definition, anonymous function (do syntax)
  - Column major, 1 based indexing
  - For newcomers, open Bachelor course and read it
  - **LABS:**
      + Check Installation \& IDE
      + Introduce `Revise.jl`
      + Toy problems

### The power of Type System \& multiple dispatch
  - Zero cost abstraction the compiler has information about types and can freely inline
    `x |> identity |> identity |> identity |> identity |> identity`
  - Why the type system is important for efficiency
  - Bad practices 
  - **LABS:**
      + Number type-tree
      + Design Interval Aritmetics (for ODES)

### Packages, environments, testing, continuous integration, dependency
  - **LABS:**
    + Create a package for interval arithmetic (github, unit tests, tomls, compat)
    + CI is important for debugging code and necessary 

### Design patterns: scoping / closure / opaque closures / interfaces / traits
  - Closure for abstraction
  - Lecture example: monitoring gradient descend.
    + function to create minibatch
    + callbacks for monitoring / early stopping / storing stuff / timed printing
  - How type system allows efficient closures (most of the time, there is no performance penalty)
  - The poor state of interfaces: traits
  - LABS
    + Small examples (GD?, Minibatching, )
    +  Performance issues ()

### Benchmarking, profiling, and performance gotchas
  - Lecture example: Static arrays (loop unrolling), 
  - How-to and not-to do benchmarking (global scope, interpolation of variables)
  - The pitfalls of global variables
  - heap / stack allocation, (mutable) struct
  - The use of profiler and visualization PProf / Flamegraphs
  - profiling memory usage
  - **LABS:**
    + fix broken examples

### Language introspection
  - Different levels of compilation
  - What AST
  - Manipulating AST
  - **LABS:**
    + Removing boundary checks
    + generate call graph of a code (TreeView.jl)

### Macros
  - Difference between function and macros
  - Macro hygiene
  - Lecture example @btime macro
  - Function approximation (sin, exp)
  - Implementing domain specific languages (Soss.jl / ModellingToolkit / NN / NiLang)
  - **LABS:**
    + Own implementation of Chain.jl
    + Macro for defining intervals or other advanced number types

### Introduction to automatic differentiation
  - For a broader context, we can use this one https://www.stochasticlifestyle.com/generalizing-automatic-differentiation-to-automatic-sparsity-uncertainty-stability-and-parallelism/
  - Adjoints
  - Forward diff and Dual numbers
  - ReverseDiff with operator overloading
  - **LABS:**
    + Implement forward diff with Dual Numbers

### Manipulating intermediate representation
  - Generated functions
  - Zygote / IR tools
  - AD as a compiler problem
  - **LABS:**
    + Implement Reverse diff

### Different levels of parallel programming (SIMD, Tasks, Threads, Processes)
  - Paralellizing stochastic gradient descend
  - **LABS:**
    + Speeding up a simple sum of an array (start with simd and work our way up)
    + embarrassingly parallel example: asteroid belt around a sun (NBody simulation)
    + PDE? discretize by space patches, DistributedArrays.jl?

### Julia for GPU programming
  - Generic concepts: map, reduce
  - Writing own kernels
  - Re-targeting compiler
  - **LABS:**
    + Interval arithmetic on GPU

### Uncertainty propagation in ODE 
  - follow: `https://mitmath.github.io/18337/lecture19/uncertainty_programming`
  - Introduce ODE
  - Solvers
  - **LABS:**
      + Plots, reciepes
      + Lotka-Volterra

### Learning  ODE  from data
  - Differentiation through ODE, Adjoints?
  - Forward Diff
  - Physics informed NN 
  - **LABS:**
    + Learning Lotka-Volterra from the data

## Resources

* https://computationalthinking.mit.edu/Fall20/
* https://github.com/mitmath/18337
* https://github.com/MichielStock/STMO
* https://laurentlessard.com/teaching/524-intro-to-optimization/