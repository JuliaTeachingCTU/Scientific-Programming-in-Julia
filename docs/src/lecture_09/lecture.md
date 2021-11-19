# Generated functions

Questions to discourse:
- What the if `Expr(:generated)` means? Who decides which branch will be taken?

```julia
@generated function gentest(x)
    return :(x + x)
end
```

```julia
julia> @macroexpand @generated function gentest(x)
           return :(x + x)
       end

:(function gentest(x)
      if $(Expr(:generated))
          return $(Expr(:copyast, :($(QuoteNode(:(x + x))))))
      else
          $(Expr(:meta, :generated_only))
          return
      end
  end)
```
- The body of the function contains a check, if compiler wants generated function `Expr(:generated)` (there is a possibility to write two versions of the function, one generated and one traditional and the compiler might choose, which one he preferes)
- the output of the function `gentest` is wrappend in `QuoteNode` (recall this means a full quotation meaning that `$(.)` is not interpolated in inserted to ensure that the code will be inserted as-is) and inserted into the AST.
- the generated function has to be created after *type-anotating* pass of the compiler

#### Notable differences to Macros
- Generated functions return expressions, similarly as macros. 
- But, the argument of macros is an expression, whereas in generated functions, it is **dummy variables**, which means
    + You can use their name in the expressions produced by the generated functions. 
    + you have access to the type information about the variables (note they are called during inferring )
- Generated functions has to be pure in the sense that they are not allowed to have side effects (for example modifying some global variables, including things like printing). The reason for this is that this can lead to unexpected errors, as you do not know, at which moment the functions will be called.
- Generated function cannot functions that has not been defined prior their execution.

Let's demonstrate everything on a version of `map` that that can be applied on `NamedTuple`s with permuted names. Recall the behavior of normal map, which works if names are the same
```julia
x = (a = 1, b = 2, c = 3)
y = (a = 4, b = 5, c = 6)
map(+, x, y)
```
but it does not work with permuted names
```julia
x = (a = 1, b = 2, c = 3)
y = (c = 6, b = 5, a = 4)
map(+, x, y)
```

How can we fix that?
The usual approach would be to iterate over the keys in named tuples as 
```julia
function permuted_map(f, x::NamedTuple{KX}, y::NamedTuple{KY}) where {KX, KY}
    @assert issubset(KX,KY)
    NamedTuple{KX}(map(k -> f(x[k], y[k]), KX))
end
```

But, can we do better? Recall that in NamedTuples, we exactly know the position of the arguments, hence we should be able to directly match the corresponding arguments without using `get`. 

Since creation (and debugging) of generated functions is difficult, we start with a single single-argument unrolled map
```julia
@generated function unrolled_map(f, x::NamedTuple{KX}) where {KX} 
    vals = [:(f(getfield(x, $(QuoteNode(k))))) for k in KX]
    :(($(vals...),))
end
```
We see inserting `Symbol` specifying the field in the NamedTuple is a bit tricky. It needs to be quoted, since `$()` which is needed to substitute `k` for its value "peels" one layer of the quoting. Compare it to `vals = [:(f(getfield(x, $(k)))) for k in KX]`

Since getting the field is awkward, we write syntactic sugar for that
```julia
_get(name, k) = :(getfield($(name), $(QuoteNode(k))))
```
with that, we proceed to a nicer two argument function which we have desired
```julia
@generated function unrolled_map(f, x::NamedTuple{KX}, y::NamedTuple{KY}) where {KX, KY} 
    @assert issubset(KX,KY)
    _get(name, k) = :(getfield($(name), $(QuoteNode(k))))
    vals = [:(f($(_get(:x, k)), $(_get(:y, k)))) for k in KX]
    :(NamedTuple{$(KX)}(($(vals...),)))
end
```
We can check that the `unrolled_map` unrolls the map and generates just needed operations
```julia
julia> @code_typed unrolled_map(+, x, y)
CodeInfo(
1 ─ %1  = Main.getfield(x, :a)::Int64
│   %2  = Main.getfield(y, :a)::Int64
│   %3  = Base.add_int(%1, %2)::Int64
│   %4  = Main.getfield(x, :b)::Int64
│   %5  = Main.getfield(y, :b)::Int64
│   %6  = Base.add_int(%4, %5)::Int64
│   %7  = Main.getfield(x, :c)::Int64
│   %8  = Main.getfield(y, :c)::Int64
│   %9  = Base.add_int(%7, %8)::Int64
│   %10 = %new(NamedTuple{(:a, :b, :c), Tuple{Int64, Int64, Int64}}, %3, %6, %9)::NamedTuple{(:a, :b, :c), Tuple{Int64, Int64, Int64}}
└──       return %10
) => NamedTuple{(:a, :b, :c), Tuple{Int64, Int64, Int64}}
```
and compare this to the same of 
```julia
@code_typed permuted_map(+, x, y)
```
which is not shown here for the sake of conciseness.

For fun, we can create a version which replaces the `Symbol` arguments directly by position numbers
```julia
@generated function unrolled_map(f, x::NamedTuple{KX}, y::NamedTuple{KY}) where {KX, KY} 
    @assert issubset(KX,KY)
    _get(name, k, KS) = :(getfield($(name), $(findfirst(k .== KS))))
    vals = [:(f($(_get(:x, k, KX)), $(_get(:y, k, KY)))) for k in KX]
    :(NamedTuple{$(KX)}(($(vals...),)))
end
```
## Example with getindex from docs





## Optionally generated functions
*Currently the @generated branch is always used. In the future, which branch is used will mostly depend on whether the JIT compiler is enabled and available, and if it's not available, then it will depend on how much we were able to compile before the compiler was taken away. So I think it will mostly be a concern for those that might need static compilation and JIT-less deployment.* [see](https://github.com/JuliaLang/julia/pull/23168)


## Contextual dispatch / Overdubbing
Imagine that you would like to give functions a different meaning if they are executed under some context. Motivation from `Cassette.jl` implementing this technique *includes dynamic code analysis (e.g. profiling, rr-style debugging, etc.), JIT compilation to new hardware/software backends, automatic differentiation, interval constraint programming, automatic parallelization/rescheduling, automatic memoization, lightweight multistage programming, graph extraction, and more.*
In theory, we can do all the above by directly modifying the code or introducing new types, but that may require a lot of coding and changing foreign libraries.

The technique we desire is called contextual dispatch, which means that under some context, we invoke a different function. The library `Casette.jl` provides a high-level api for overdubbing, but it is by no means interesting to see, how it works, as it shows, how we can "interact" with the lowered code before the code is typed.

```julia
world = Base.get_world_counter()
sigtypes = (typeof(+), Int, Int)
sigtypes = (typeof(sin), Int)

function retrieve_code_info(sigtypes, world = Base.get_world_counter())
  S = Tuple{map(s -> Core.Compiler.has_free_typevars(s) ? typeof(s.parameters[1]) : s, sigtypes)...}
  _methods = Base._methods_by_ftype(S, -1, world)
  isempty(_methods) && @error("method $(sigtypes) does not exist, may-be run it once")
  type_signature, raw_static_params, method = _methods[1] # method is the same as we would get by invoking methods(+, (Int, Int)).ms[1]  

  # this provides us with withe CodeInfo
  method_instance = Core.Compiler.specialize_method(method, type_signature, raw_static_params, false)
  code_info = Core.Compiler.retrieve_code_info(method_instance)
end
```
[details of code info](https://docs.julialang.org/en/v1/devdocs/ast/#CodeInfo)


```julia
foo(x) = 5x + 3
foo(x,y) = x * y + sin(x)
foo(1.0,1.0)
sigtypes = (typeof(foo), Float64, Float64)
sigtypes = (typeof(foo), Float64, Float64)
```

```julia
lowered_code = retrieve_code_info((typeof(foo), Float64))
lowered_code = retrieve_code_info((typeof(foo), Float64, Float64))
lowered_code.code        # contains expression of individual lines of code
lowered_code.slotnames   # contains names of corresponding variables
```

Let's list for which expressions we have rrules in ChainRules
```julia
using ChainRules, ChainRulesCore
rename_args(ex::Expr, primal_names, lowered_fun) = Expr(ex.head, rename_args(ex.args, primal_names, lowered_fun))

function rename_args(args::Vector, primal_names, lowered_fun)
   map(args) do a 
    rename_args(a, primal_names, lowered_fun)
  end
end

function rename_args(a, primal_names, lowered_fun)
  if a isa Core.SlotNumber
    return(lowered_fun.slotnames[a.id])
  elseif a isa Core.SSAValue
    return(primal_names[a.id])
  else
    return(a)
  end
end

rename_args(a::Core.ReturnNode, primal_names, lowered_fun) = Core.ReturnNode(rename_args(a.val, primal_names, lowered_fun))

function generate_grad(lowered_code)
  primal_names = Dict{Int,Symbol}()
  primals = []
  pullbacks = []
  pullback_names = Dict{Int,Any}()
  rename_args(ex) = rename_args(ex, primal_names, lowered_code)
  output_id = -1

  for(exno, ex) in enumerate(lowered_code.code)
    if ex isa Core.ReturnNode
      output_id = ex.val.id
      continue
    end

    if !hasproperty(ex, :head) || ex.head != :call
      push!(primals, rename_args(ex))
      continue
    end

    ex.head != :call && return(rename_args(ex))
    fun_T = :(typeof($(ex.args[1]))) |> eval
    args_T = map(i -> Number, ex.args[2:end])
    _methods = Base._methods_by_ftype(Tuple{typeof(rrule), fun_T, args_T...}, -1, world)
    isempty(_methods) && return(rename_args(ex)) # If the method does not exist, I need to dive_in
    renamed_args = rename_args(ex.args)

    primal   = Symbol("r", exno)
    pullback = Symbol("∂", exno)
    rr = Symbol("rr",exno)
    primal_names[exno] = primal
    pullback_names[exno] = (pullback, renamed_args...)

    # let's work out the pullback
    push!(primals, :($(rr) = rrule($(renamed_args...))))
    push!(primals, :($(primal) = $(rr)[1]))
    push!(primals, :($(pullback) = $(rr)[2]))
  end

  #Let's get the name of the output variable and iniate the pullback
  primal_out = Symbol(:r, output_id) 
  pullback_arg = Symbol(:∂r, output_id) 
  created_grads = Indices{Symbol}()
  for line_no in sort(collect(keys(pullback_names)), rev = true)
    pullfun  = pullback_names[line_no][1]
    pullout = Symbol(:Δ, line_no)
    p = Symbol(:∂r, line_no) 
    push!(pullbacks, :($(pullout) = $(pullfun)($(p))))
    for (i, x) in (enumerate(pullback_names[line_no][3:end]))
      o = Symbol(:∂, x) 
      if o ∈ created_grads
        push!(pullbacks, :($(o) += $(pullout)[$(i+1)]))
      else
        push!(pullbacks, :($(o) = $(pullout)[$(i+1)]))
        insert!(created_grads, o)
      end
    end
  end

  #add gradients with respect to arguments
  argnames = lowered_code.slotnames[2:end]
  ∂args = map(s -> Symbol(:∂, s), argnames)

  quote
    function ∂foo($(argnames...))
      $(primals...)
      function pullback($(pullback_arg))
        $(pullbacks...)
        return(NoTangent(), $(∂args...))
      end
      return($(primal_out), pullback)
    end
  end |> Base.remove_linenums!
end

p, back = ∂foo(1, 1)
p ≈ foo(1,1)
back(1)
```

### some thoughts
ri is the variable
∂i is a function to which I need to pass the ∂ri to get the gradinets 



# TODO

* show constant gradient of linear function in LLVM
