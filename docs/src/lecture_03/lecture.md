# Design patterns: good practices and structured thinking

Design guiding principles:
- SOLID: Single Responsibility, Open/Closed, Liskov Substitution, Interface
- Segregation, Dependency Inversion
- DRY: Don't Repeat Yourself
- KISS: Keep It Simple, Stupid!
- POLA: Principle of Least Astonishment
- YAGNI: You Aren't Gonna Need It (overengineering)
- POLP: Principle of Least Privilege 

Julia does not fit into any methodological classes like *object-oriented* or *functional* programming. The key concept of julia is the *multiple dispatch*, which has *zero* runtime cost.

Many popular design concepts from other languages are solved using this simple principle. Multiple dispatch allows to 

- composition vs. inheritance (L02?)
- generalization problem
- reusability (composition, @forward)
- access restrictions (getter functions, redefining getproperty)
    - closures - functions with states(), accepting 
- piracy?
- traits, multiple-inheritance


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

### Implementation of closures in julia: documentation

Closure is a record storing a function together with an environment. The environment is a mapping associating each *free* variable of the function (variables that are used locally, but defined in an enclosing scope) with the value or reference to which the name was bound when the closure was created.

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

Note that the structure ##1 is not directly accessible. Allowing access restriction. 

Usage of closures:
- callbacks: the function can also modify the enclosed variable.
- abstraction: partial evaluation 
- can be used to imitate objects: 
https://stackoverflow.com/questions/39133424/how-to-create-a-single-dispatch-object-oriented-class-in-julia-that-behaves-l/39150509#39150509

!!! theorem "Beware: Performance of captured variables"
    Inference of types may be difficult in closures:
    https://github.com/JuliaLang/julia/issues/15276    


### Expression Problem 
Matrix of methods/types(data-structures)

| data \ methods | find_food | eat! |  |  | new |
| --- | ---- | ---- | --- | ---- | -- |
| Wolf |  | | | | |
| Sheep | | | | | |
| Grass | | | | | |
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
