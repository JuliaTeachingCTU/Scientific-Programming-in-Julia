# Syntax

## Elementary syntax: Matlab heritage

Very much like matlab:

- indexing from  1
- array as first-class ```A=[1 2 3]```


::: tip Useful links

- Cheat sheet: https://cheatsheets.quantecon.org/
- Introduction: https://juliadocs.github.io/Julia-Cheat-Sheet/

:::


### Arrays are first-class citizens

Many design choices were motivated considering matrix arguments:

- ``` x *= 2``` is implemented as ```x = x*2``` causing new allocation (vectors).

The reason is consistency with matrix operations: ```A *= B``` works as ```A = A*B```.

### Broadcasting operator

Julia generalizes matlabs ```.+``` operation to general use for any function. 

```julia
a = [1 2 3]
sin.(a)
f(x)=x^2+3x+8
f.(a)
```

Solves the problem of inplace multiplication

- ``` x .*= 2``` 

The ```a.+b``` syntax is a syntactic sugar for ```broadcast(+,a,b)```.

The special meaning of the dot is that they will be fused into a single call:

- ```f.(g.(x .+ 1))``` is treated by Julia as ```broadcast(x -> f(g(x + 1)), x)```. 
- An assignment ```y .= f.(g.(x .+ 1))``` is treated as in-place operation ```broadcast!(x -> f(g(x + 1)), y, x)```.

The same logic works for lists, tuples, etc.


## Functional roots of Julia

Function is a first-class citizen.

Repetition of functional programming:

```julia 
function mymap(f::Function,a::AbstractArray)
    b = similar(a)
    for i in eachindex(a)
        b[i]=f(a[i])
    end
    b
end
```

Allows for anonymous functions:

```julia
mymap(x->x^2+2,[1.0,2.0])
```

Function properties:

- Arguments are passed by reference (change of mutable inputs inside the function is visible outside)
- Convention: function changing inputs have a name ending by "!" symbol
- return value 
  -  the last line of the function declaration, 
  - ```return``` keyword
- zero cost abstraction

### Different style of writing code

Definitions of multiple small functions and their composition (recall ```fsum``` from the teaser)

```julia
fsum(x) = x
fsum(x,p...) = x+fsum(p...)
```

a single methods may not be sufficient to understand the full algorithm. In procedural language, you may write:

```matlab
function out=fsum(x,varargin)
    if nargin==1
        out=x;
    else
        out = x + fsum(varargin{:});
    end
```

The need to build intuition for function composition.

Dispatch is easier to optimize by the compiler.


## Operators are functions

| operator        | function name |
| ---             | ---           |
| [A B C ...]     | hcat          |
| [A; B; C; ...]  | vcat          |
| [A B; C D; ...] | hvcat         |
| A'              | adjoint       |
| A[i]            | getindex      |
| A[i] = x	      | setindex!     |
| A.n             | getproperty   |
| A.n = x         | setproperty!  |

```julia
struct Foo end

Base.getproperty(a::Foo, x::Symbol) = x == :a ? 5 : error("does not have property $(x)")
```

Can be redefined and overloaded for different input types. The ```getproperty``` method can define access to the memory structure.

What did Measurements need to overload?

## Reproducible research

Think about a code that was written some time ago. To run it, you often need to be able to have the same version of the language it was written for. 

- **Standard way** language freezes syntax and guarantees some back-ward compatibility (Matlab), which prevents future improvements

- **Julia approach** allows easy recreation of the *environment* in which the code was developed. Every project (e.g. directory) can have its own environment

::: tip Environment

Is an independent set of packages that can be local to an individual project or shared and selected by name.

:::

::: tip Package

A package is a source tree with a standard layout providing functionality that can be reused by other Julia projects.

:::

This allows  Julia to be a  rapidly evolving ecosystem with frequent changes due to:

- built-in package manager
- switching between multiple versions of packages


### Package manager

- implemented by Pkg.jl
- source tree have their structure defined by a convention
- have its own mode in REPL
- allows adding packages for using (```add```) or development (```dev```)
- supporting functions for creation (```generate```) and activation (```activate```) and many others
