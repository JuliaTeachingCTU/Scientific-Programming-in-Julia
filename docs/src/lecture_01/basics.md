# Syntax

## Elementary syntax: Matlab heritage
Very much like matlab:
- indexing from  1
- array as first-class ```A=[1 2 3]```

Cheat sheet: https://cheatsheets.quantecon.org/

Introduction: https://juliadocs.github.io/Julia-Cheat-Sheet/


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



## Functional roots of Julia
Function is a first-class citizen.

Repetition of functional programming:
```julia 
function mymap(f::Function,a::AbstractArray)
    b = similar(a)
    for i=1:length(a)
        b[i]=f(a[i])
    end
    b
end
```

Allows for anonymous functions:
```
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

Definitions of multiple small functions and their composition
```
fsum(x) = x
fsum(x,p...) = x+fsum(p[1],p[2:end]...)
```
a single methods may not be sufficient to understand the full algorithm. In procedural language, you may write:
```matlab
function out=fsum(x,varargin)
if nargin==2 # TODO: better treatment
    out=x
else
    out = fsum(varargin{1},varargin{2:end})
end
```
The need to build intuition for function composition.

Dispatch is easier to optimize by the compiler.


## Operators are functions

| operator | function name |
| --- | --- |
| [A B C ...]	| hcat |
| [A; B; C; ...]	|vcat|
| [A B; C D; ...]	|hvcat|
| A'|	adjoint|
| A[i]	|getindex|
| A[i] = x	|setindex!|
| A.n|	getproperty|
| A.n = x	|setproperty!|


Can be redefined and overloaded for different input types. The ```getproperty``` method can define access to the memory structure.

## Broadcasting revisited

The ```a.+b``` syntax is a syntactic sugar for ```broadcast(+,a,b)```.

The special meaning of the dot is that they will be fused into a single call:

- ```f.(g.(x .+ 1))``` is treated by Julia as ```broadcast(x -> f(g(x + 1)), x)```. 
- An assignment ```y .= f.(g.(x .+ 1))``` is treated as in-place operation ```broadcast!(x -> f(g(x + 1)), y, x)```.

The same logic works for lists, tuples, etc.
