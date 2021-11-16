# Generated functions

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
- The body of the function contains a check, if compiler wants generated function `Expr(:generated)` 
- the output of the function `gentest` is wrappend in `QuoteNode` (recall this means a full quotation meaning that `$(.)` is not interpolated in) and inserted into the AST.
- 

Questions to discourse:
- What the if `Expr(:generated)` means? Who decides which branch will be taken?
- At which stage of compilation the generation function takes place? It is probably when the compiler is performing type inference and it is looking for a specialization of a function for a given set of arguments.


#### Notable differences to Macros
- Generated functions return expressions, similarly as macros. 
- In macros, the argument was an expression. In generated functions, the argument is syntactically variables, but you do not have access to their values. You can use their name in the expressions produced by the generated functions. Importantly, the generated functions have access to the type information about the variables (note they are called during inferring )
- Generated functions has to be pure in the sense that they are not allowed to have side effects (for example modifying some global variables, including things like printing). The reason for this is that this can lead to unexpected errors, as you do not know, at which moment the functions will be called


Let's write a version of `map` that would apply the function `f` on arguments with corresponding names.
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
Approach 1:
```julia
function permuted_map(f, x::NamedTuple{KX}, y::NamedTuple{KY}) where {KX, KY}
    @assert issubset(KX,KY)
    NamedTuple{KX}(map(k -> f(x[k], y[k]), KX))
end
```

Let's start with simple single-argument unrolled map
```julia
@generated function unrolled_map(f, x::NamedTuple{KX}) where {KX} 
    vals = [:(f(getfield(x, $(QuoteNode(k))))) for k in KX]
    :(vcat($(vals...)))
end
```

We can see that obtaining the name of a varible is a little bit difficult, so let's write a syntactic sugar
```julia
_get(name, k) = :(getfield($(name), $(QuoteNode(k))))
```
and with that, the code will look nicer
```julia
@generated function unrolled_map(f, x::NamedTuple{KX}, y::NamedTuple{KY}) where {KX, KY} 
    @assert issubset(KX,KY)
    _get(name, k) = :(getfield($(name), $(QuoteNode(k))))
    vals = [:(f($(_get(:x, k)), $(_get(:y, k)))) for k in KX]
    :(NamedTuple{$(KX)}(($(vals...),)))
end
```

```julia
@generated function unrolled_map(f, x::NamedTuple{KX}, y::NamedTuple{KY}) where {KX, KY} 
    @assert issubset(KX,KY)
    _get(name, k, KS) = :(getfield($(name), $(findfirst(k .== KS))))
    vals = [:(f($(_get(:x, k, KX)), $(_get(:y, k, KY)))) for k in KX]
    :(NamedTuple{$(KX)}(($(vals...),)))
end
```


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
