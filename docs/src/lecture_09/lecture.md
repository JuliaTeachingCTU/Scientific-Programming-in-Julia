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
foo(x,y) = x * y + sin(x)
foo(1.0,1.0)
sigtypes = (typeof(foo), Float64, Float64)
```

```julia
lowered_foo = retrieve_code_info((typeof(foo), Float64, Float64))
lowered_foo.code        # contains expression of individual lines of code
lowered_foo.slotnames   # contains names of corresponding variables
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


primal_names = Dict{Int,Symbol}()
primals = []
pullback_names = Dict{Int,Any}()
rename_args(ex) = rename_args(ex, primal_names, lowered_foo)
output_id = -1

for(exno, ex) in enumerate(lowered_foo.code)
  if ex isa lowered_foo.code[end]
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
  push!(primals, :($(pullback) = $(rr)[2])])
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
      push!(pullbacks, :($(o) += $(pullout)[$(i)]))
    else
      push!(pullbacks, :($(o) = $(pullout)[$(i)]))
      insert!(created_grads, o)
    end
  end
end

quote
  function ∂foo(x, y)
    $(primals...)
    function pullback($(Δ))
      $(pullbacks...)
    end
    return(())
  end
end |> Base.remove_linenums!

∂foo(1, 1)[1] ≈ foo(1,1)
```

### some thoughts
ri is the variable
∂i is a function to which I need to pass the ∂ri to get the gradinets 



## Zygote internals

```julia
function pullback(f, args...)
  y, back = _pullback(f, args...)
  y, Δ -> tailmemaybe(back(Δ))
end

function gradient(f, args...)
  y, back = pullback(f, args...)
  grad = back(sensitivity(y))
  isnothing(grad) ? nothing : map(_project, args, grad)
end

_pullback(f, args...) = _pullback(Context(), f, args...)

@generated function _pullback(ctx::AContext, f, args...)
  # Try using ChainRulesCore
  if is_kwfunc(f, args...)
    # if it is_kw then `args[1]` are the keyword args, `args[2]` is actual function
    cr_T = Tuple{ZygoteRuleConfig{ctx}, args[2:end]...}
    chain_rrule_f = :chain_rrule_kw
  else
    cr_T = Tuple{ZygoteRuleConfig{ctx}, f, args...}
    chain_rrule_f = :chain_rrule
  end

  hascr, cr_edge = has_chain_rrule(cr_T)
  hascr && return :($chain_rrule_f(ZygoteRuleConfig(ctx), f, args...))

  # No ChainRule, going to have to work it out.
  T = Tuple{f,args...}
  ignore_sig(T) && return :(f(args...), Pullback{$T}(()))

  g = try
    _generate_pullback_via_decomposition(T)
  catch e
    rethrow(CompileError(T,e))
  end
  g === nothing && return :(f(args...), Pullback{$T}((f,)))
  meta, forw, _ = g
  argnames!(meta, Symbol("#self#"), :ctx, :f, :args)
  forw = varargs!(meta, forw, 3)
  # IRTools.verify(forw)
  forw = slots!(pis!(inlineable!(forw)))
  # be ready to swap to using chainrule if one is declared
  cr_edge != nothing && edge!(meta, cr_edge)
  return update!(meta.code, forw)
end
```
## Source Code Transformation

The most recent approach to Reverse Mode AD is **_Source-to-Source_**
transformation adopted by packages like **_JAX_** and **_Zygote.jl_**.
Transforming code promises to eliminate the problems of tracing-based AD.
`Tracked` types are not needed anymore, which reduces memory usage, promising
significant speedups. Additionally, the reverse pass becomes a *compiler
problem*, which makes it possible to leverage highly optimized compilers like
LLVM.

Source-to-source AD uses meta-programming to produce `rrule`s for any function
that is a composition of available `rrule`s. The code for `foo`
```@example lec08
foo(x) = h(g(f(x)))

f(x) = x^2
g(x) = sin(x)
h(x) = 5x
nothing # hide
```
is transformed into
```julia eval=false
function rrule(::typeof(foo), x)
    a, Ja = rrule(f, x)
    b, Jb = rrule(g, a)
    y, Jy = rrule(h, b)

    function dfoo(Δy)
        Δb = Jy(Δy)
        Δa = Jb(Δb)
        Δx = Ja(Δa)
        return Δx
    end
    
    return y, dfoo
end
```
For this simple example we can define the three `rrule`s by hand:
```@example lec08
rrule(::typeof(f), x) = f(x), Δ -> 2x*Δ
rrule(::typeof(g), x) = g(x), Δ -> cos(x)*Δ
rrule(::typeof(h), x) = h(x), Δ -> 5*Δ
```
Remember that this is a very artificial example. In real AD code you would
overload functions like `+`, `*`, etc, such that you don't have to define a
`rrule` for something like `5x`.

In order to transform our functions safely we will make use of `IRTools.jl`
(*Intermediate Representation Tools*) which provide some convenience functions
for inspecting and manipulating code snippets. The IR for `foo` looks like this:
```@example lec08
using IRTools: @code_ir, evalir
ir = @code_ir foo(2.)
```
```@setup lec08
msg = """
ir = 1: (%1, %2)                 ## rrule(foo, x)
       %3 = Main.f(%2)           ##   a = f(x)
       %4 = Main.g(%3)           ##   b = g(a)
       %5 = Main.h(%4)           ##   y = h(b)
       return %5                 ##   return y
"""
```
Variable names are replaced by `%N` and each function gets is own line.
We can evalulate the IR (to actually run it) like this
```@example lec08
evalir(ir, nothing, 2.)
```
As a first step, lets transform the function calls to `rrule` calls.  For
this, all we need to do is iterate through the IR line by line and replace each
statement with `(Main.rrule)(Main.func, %N)`, where `Main` just stand for the
gobal main module in which we just defined our functions.
But remember that the `rrule` returns
the value `v` *and* the pullback `J` to compute the gradient. Just
replacing the statements would alter our forward pass. Instead we can insert
each statement *before* the one we want to change. Then we can replace the the
original statement with `v = rr[1]` to use only `v` and not `J` in the
subsequent computation.
```@example lec08
using IRTools
using IRTools: xcall, stmt

xgetindex(x, i...) = xcall(Base, :getindex, x, i...)

ir = @code_ir foo(2.)
pr = IRTools.Pipe(ir)

for (v,statement) in pr
    ex = statement.expr
    rr = xcall(rrule, ex.args...)
    # pr[v] = stmt(rr, line=ir[v].line)
    vJ = insert!(pr, v, stmt(rr, line = ir[v].line))
    pr[v] = xgetindex(vJ,1)
end
ir = IRTools.finish(pr)
#
#msg = """
#ir = 1: (%1, %2)                          ## rrule(foo, x)
#       %3 = (Main.rrule)(Main.f, %2)      ##   ra = rrule(f,x)
#       %4 = Base.getindex(%3, 1)          ##   a  = ra[1]
#       %5 = (Main.rrule)(Main.g, %4)      ##   rb = rrule(g,a)
#       %6 = Base.getindex(%5, 1)          ##   b  = rb[1]
#       %7 = (Main.rrule)(Main.h, %6)      ##   ry = rrule(h,b)
#       %8 = Base.getindex(%7, 1)          ##   y  = ry[1]
#       return %8                          ##   return y
#"""
#println(msg)
```
Evaluation of this transformed IR should still give us the same value
```@example lec08
evalir(ir, nothing, 2.)
```

The only thing that is left to do now is collect the `Js` and return
a tuple of our forward value and the `Js`.
```@example lec08
using IRTools: insertafter!, substitute, xcall, stmt

xtuple(xs...) = xcall(Core, :tuple, xs...)

ir = @code_ir foo(2.)
pr = IRTools.Pipe(ir)
Js = IRTools.Variable[]

for (v,statement) in pr
    ex = statement.expr
    rr = xcall(rrule, ex.args...)  # ex.args = (f,x)
    vJ = insert!(pr, v, stmt(rr, line = ir[v].line))
    pr[v] = xgetindex(vJ,1)

    # collect Js
    J = insertafter!(pr, v, stmt(xgetindex(vJ,2), line=ir[v].line))
    push!(Js, substitute(pr, J))
end
ir = IRTools.finish(pr)
# add the collected `Js` to `ir`
Js  = push!(ir, xtuple(Js...))
# return a tuple of the last `v` and `Js`
ret = ir.blocks[end].branches[end].args[1]
IRTools.return!(ir, xtuple(ret, Js))
ir
#msg = """
#ir = 1: (%1, %2)                          ## rrule(foo, x)
#       %3 = (Main.rrule)(Main.f, %2)      ##   ra = rrule(f,x)
#       %4 = Base.getindex(%3, 1)          ##   a  = ra[1]
#       %5 = Base.getindex(%3, 2)          ##   Ja = ra[2]
#       %6 = (Main.rrule)(Main.g, %4)      ##   rb = rrule(g,a)
#       %7 = Base.getindex(%6, 1)          ##   b  = rb[1]
#       %8 = Base.getindex(%6, 2)          ##   Jb = rb[2]
#       %9 = (Main.rrule)(Main.h, %7)      ##   ry = rrule(h,b)
#       %10 = Base.getindex(%9, 1)         ##   y  = ry[1]
#       %11 = Base.getindex(%9, 2)         ##   Jy = ry[2]
#       %12 = Core.tuple(%5, %8, %11)      ##   Js = (Ja,Jb,Jy)
#       %13 = Core.tuple(%10, %12)         ##   rr = (y, Js)
#       return %13                         ##   return rr
#"""
#println(msg)
```
The resulting IR can be evaluated to the forward pass value and the Jacobians:
```@repl lec08
(y, Js) = evalir(ir, foo, 2.)
```
To compute the derivative given the tuple of `Js` we just need to compose them
and set the initial gradient to one:
```@repl lec08
reduce(|>, Js, init=1)  # Ja(Jb(Jy(1)))
```
The code for transforming the IR as described above looks like this.
```@example lec08
function transform(ir, x)
    pr = IRTools.Pipe(ir)
    Js = IRTools.Variable[]
    
    # loop over each line in the IR
    for (v,statement) in pr
        ex = statement.expr
        # insert the rrule
        rr = xcall(rrule, ex.args...)  # ex.args = (f,x)
        vJ = insert!(pr, v, stmt(rr, line = ir[v].line))
        # replace original line with f(x) from rrule
        pr[v] = xgetindex(vJ,1)
    
        # save jacobian in a variable
        J = insertafter!(pr, v, stmt(xgetindex(vJ,2), line=ir[v].line))
        # add it to a list of jacobians
        push!(Js, substitute(pr, J))
    end
    ir = IRTools.finish(pr)
    # add the collected `Js` to `ir`
    Js  = push!(ir, xtuple(Js...))
    # return a tuple of the foo(x) and `Js`
    ret = ir.blocks[end].branches[end].args[1]
    IRTools.return!(ir, xtuple(ret, Js))
    return ir
end

xgetindex(x, i...) = xcall(Base, :getindex, x, i...)
xtuple(xs...) = xcall(Core, :tuple, xs...)
nothing # hide
```
Now we can write a general `rrule` that can differentiate any function
composed of our defined `rrule`s
```@example lec08
function rrule(f, x)
    ir = @code_ir f(x)
    ir_derived = transform(ir,x)
    y, Js = evalir(ir_derived, nothing, x)
    df(Δ) = reduce(|>, Js, init=Δ)
    return y, df
end


reverse(f,x) = rrule(f,x)[2](one(x))
nothing # hide
```
Finally, we just have to use `reverse` to compute the gradient
```@example lec08
plot(-2:0.1:2, foo, label="f(x) = 5sin(x^2)", lw=3)
plot!(-2:0.1:2, x->10x*cos(x^2), label="Analytic f'", ls=:dot, lw=3)
plot!(-2:0.1:2, x->reverse(foo,x), label="Dual Forward Mode f'", lw=3, ls=:dash)
```

---
- Efficiency of the forward pass becomes essentially a compiler problem
- If we define specialized rules we will gain performance
---

# Performance Forward vs. Reverse

This section compares the performance of three different, widely used Julia AD
systems `ForwardDiff.jl` (forward mode), `ReverseDiff.jl` (tracing-based
reverse mode), and `Zygote.jl` (source-to-source reverse mode), as well as JAX
forward/reverse modes.

As a benchmark function we can compute the Jacobian of $f:\mathbb R^N
\rightarrow \mathbb R^M$ with respect to $\bm x$.
In the benchmark we test various different values of $N$ and $M$ to show the
differences between the backends.
```math
f(\bm x) = (\bm W \bm x + \bm b)^2
```

```@setup lec08
using DataFrames
using DrWatson
using Glob


julia_res = map(glob("julia-*.txt")) do fname
    d = parse_savename(replace(fname, "julia-"=>""))[2]
    @unpack N, M = d
    lines = open(fname) |> eachline
    map(lines) do line
        s = split(line, ":")
        backend = s[1]
        time = parse(Float32, s[2]) / 10^6
        (backend, time, "$(N)x$(M)")
    end
end

jax_res = map(glob("jax-*.txt")) do fname
    d = parse_savename(replace(fname, "jax-"=>""))[2]
    @unpack N, M = d
    lines = open(fname) |> eachline
    map(lines) do line
        s = split(line, ":")
        backend = s[1]
        time = parse(Float32, s[2]) * 10^3
        (backend, time, "$(N)x$(M)")
    end
end

res = vcat(julia_res, jax_res)

df = DataFrame(reduce(vcat, res))
df = unstack(df, 3, 1, 2)
ns = names(df)
ns[1] = "N x M"
rename!(df, ns)
df = DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])

ns = df[1,:] |> values |> collect
rename!(df, ns)
```
```@example lec08
df[2:end,:] # hide
```

# TODO

* show constant gradient of linear function in LLVM
