### Design patterns: scoping / closure / opaque closures / interfaces / traits
  - Anonymous functions
  - Closures for abstraction
  - Lecture example: monitoring gradient descend.
    + function to create minibatch
    + callbacks for monitoring / early stopping / storing stuff / timed printing
  - How type system allows efficient closures (most of the time, there is no performance penalty)
  - do syntax for readability
  - The poor state of interfaces: traits
  - LABS
    + Small examples (GD?, Minibatching, )
    +  Performance issues ()

### Implementation of closeres in julia: documentation

```
function adder(x)
    return y->x+y
end
```
is lowered to (roughly):

```
struct ##1{T}
    x::T
end

(_::##1)(y) = _.x + y

function adder(x)
    return ##1(x)
end
```

### Beware: Performance of captured variables
 - https://github.com/JuliaLang/julia/issues/15276    

### Performance gotcha of global variables (make them const)

### Expression Problem 
Matrix of methods/types(data-structures)

| data \ methods | add | scalarmult | vecmult | matmul | new |
| --- | ---- | ---- | --- | ---- | -- |
| Full |  | | | | |
| Sparse | | | | | |
| diagonal | | | | | |
| SVD | | | | | |
| new |

We want to multiply matrices of different types.

Matmul make sense for all other types! matmul(Full,Full), matmul(Sparse,SVD)

OOP = define classes of matrices (maybe inheriting from abstract class Matrix)
FP = define operations "add", "scalarmult", etc.

Solutions:
1. multiple-dispatch = julia
2. open classes (monkey patching) = add methods to classes on the fly
3. visitor pattern = partial fix for OOP [extended visitor pattern using dynamic_cast]

Julia multiple dispatch:
 - special cases: 
   + method A^T*A: straightforward in general, specialized for SVD.
   + diagonal: 
   + multiplication by a permutation matrix

### Subtyping, Unions

 - the power of subtyping: Array{Float64} </: Array{Real}

### Traits = multiple inheritance

https://github.com/JuliaLang/julia/issues/2345#issuecomment-54537633
https://github.com/andyferris/Traitor.jl
https://github.com/mauro3/SimpleTraits.jl
https://github.com/tk3369/BinaryTraits.jl
