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


### Performance of captured variables
 - https://github.com/JuliaLang/julia/issues/15276    

### Performance gotcha of global variables (make them const)

### Traits

https://github.com/JuliaLang/julia/issues/2345#issuecomment-54537633
https://github.com/andyferris/Traitor.jl
https://github.com/mauro3/SimpleTraits.jl
https://github.com/tk3369/BinaryTraits.jl