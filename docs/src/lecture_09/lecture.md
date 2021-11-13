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
