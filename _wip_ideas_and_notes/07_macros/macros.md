### Macros
  - Difference between function and macros
  - Macro hygiene
  - Lecture example @btime macro
  - Function approximation (sin, exp)
  - Implementing domain specific languages (Soss.jl / ModellingToolkit / NN / NiLang)
  - **LABS:**
    + Own implementation of Chain.jl
    + Macro for defining intervals or other advanced number types



An example of a definition os a macro `mythreads` which uses **PARTR** backend to achieve composability. The macro definition was taken from `https://github.com/JuliaLang/julia/pull/35003`.

```julia
macro mythreads(expr::Expr)
    @assert expr.head == :for
    @assert length(expr.args) == 2
    @assert length(expr.args[1].args) == 2
    loopvar   = expr.args[1].args[1]
    iter      = expr.args[1].args[2]
    loop_body = expr.args[2]
    rng = gensym(:rng)
    quote
        @sync for $rng in $(Iterators.partition)($iter, length($iter) ÷ Threads.nthreads())
            Threads.@spawn begin
                for $loopvar in $rng
                    $loop_body
                end
            end
        end
    end |> esc
end
```


Exfiltrate macro from  https://github.com/JuliaDebug/Infiltrator.jl os also really beatiful
```julia
"""
    @exfiltrate
Assigns all local variables into the global storage.
"""
macro exfiltrate()
  quote
    for (k, v) in Base.@locals
      try
        Core.eval(getfield($(store), :store), Expr(:(=), k, QuoteNode(v)))
      catch err
        println(stderr, "Assignment to store variable failed.")
        Base.display_error(stderr, err, catch_backtrace())
      end
    end
  end
end
```

### For labs

We can consider some examples from 
https://github.com/MikeInnes/Lazy.jl/blob/master/src/macros.jl


### Some material to consider

https://www.juliabloggers.com/julia-macros-for-beginners/