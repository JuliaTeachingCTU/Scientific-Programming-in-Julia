# Lab 09 - Generated Functions & IR

In this lab you will practice two advanced meta programming techniques:

* _**Generated functions**_ can help you write specialized code for certain
  kinds of parametric types with more flexibility and/or less code.
* _**IRTools.jl**_ is a package that simplifies the manipulation of lowered and
  typed Julia code

```@setup lab09
using BenchmarkTools
```

## `@generate`d Functions

Remember the three most important things about generated functions:
* They return *quoted expressions* (like macros).
* You have access to type information of your input variables.
* They have to be _**pure**_

### A faster `polynomial`

Throughout this course we have come back to our `polynomial` function which
evaluates a polynomial based on the Horner schema. Below you can find a version
of the function that operates on a tuple of length $N$.
```@example lab09
function polynomial(x, p::NTuple{N}) where N
    acc = p[N]
    for i in N-1:-1:1
        acc = x*acc + p[i]
    end
    acc
end
nothing # hide
```
Julia has its own implementation of this function called `evalpoly`. If we
compare the performance of our `polynomial` and Julia's `evalpoly` we can
observe a pretty big difference:
```@repl lab09
x = 2.0
p = ntuple(float,20);

@btime polynomial($x,$p)
@btime evalpoly($x,$p)
```
Julia's implementation uses a generated function which specializes on different
tuple lengths (i.e. it *unrolls* the loop) and eliminates the (small) overhead
of looping over the tuple. This is possible, because the length of the tuple is
known during compile time. You can check the difference between `polynomial`
and `evalpoly` yourself via the introspectionwtools you know - e.g.
`@code_lowered`.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Rewrite the `polynomial` function as a generated function with the signature
```
genpoly(x::Number, p::NTuple{N}) where N
```
**Hints:**
* Remember that you have to generate a quoted expression inside your generated
  function, so you will need things like `:($expr1 + $expr2)`.
* You can debug the expression you are generating by omitting the `@generated`
  macro from your function.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab09
@generated function genpoly(x, p::NTuple{N}) where N
    ex = :(p[$N])
    for i in N-1:-1:1
        ex = :(x*$ex + p[$i])
    end
    ex
end
nothing # hide
```
```@raw html
</p></details>
```
You should get the same performance as `evalpoly` (and as `@poly` from Lab 7 with
the added convenience of not having to spell out all the coefficients in your code
like: `p = @poly 1 2 3 ...`).
```@repl lab09
@btime genpoly($x,$p)
```



### Fast, Static Matrices

Another great example that makes heavy use of generated functions are *static
arrays*. A static array is an array of fixed size which can be implemented via
an `NTuple`. This means that it will be allocated on the stack, which can buy
us a lot of performance for smaller static arrays. We define a
`StaticMatrix{T,C,R,L}` where the paramteric types represent the matrix element
type `T` (e.g. `Float32`), the number of rows `R`, the number of columns `C`,
and the total length of the matrix `L=C*R` (which we need to set the size of
the `NTuple`).
```@example lab09
struct StaticMatrix{T,R,C,L} <: AbstractArray{T,2}
    data::NTuple{L,T}
end

function StaticMatrix(x::AbstractMatrix{T}) where T
    (R,C) = size(x)
    StaticMatrix{T,R,C,C*R}(x |> Tuple)
end
nothing # hide
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
As a warm-up, overload the `Base` functions `size`, `length`,
`getindex(x::StaticMatrix,i::Int)`, and `getindex(x::Solution,r::Int,c::Int)`.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab09
Base.size(x::StaticMatrix{T,R,C}) where {T,R,C} = (R,C)
Base.length(x::StaticMatrix{T,R,C,L}) where {T,R,C,L} = L
Base.getindex(x::StaticMatrix, i::Int) = x.data[i]
Base.getindex(x::StaticMatrix{T,R,C}, r::Int, c::Int) where {T,R,C} = x.data[R*(c-1) + r]
```
```@raw html
</p></details>
```

You can check if everything works correctly by comparing to a normal `Matrix`:
```@repl lab09
x = rand(2,3)
x[1,2]
a = StaticMatrix(x)
a[1,2]
```


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Overload matrix multiplication between two static matrices
```julia
Base.:*(x::StaticMatrix{T,K,M},y::StaticMatrix{T,M,N})
```
with a generated function that creates an expression without loops.  Below you
can see an example for an expression that would be generated from multiplying
two $2\times 2$ matrices.
```julia
:(StaticMatrix{T,2,2,4}((
    (x[1,1]*y[1,1] + x[1,2]*y[2,1]),
    (x[2,1]*y[1,1] + x[2,2]*y[2,1]),
    (x[1,1]*y[1,2] + x[1,2]*y[2,2]),
    (x[2,1]*y[1,2] + x[2,2]*y[2,2])
)))
```

**Hints:**

* You can get output like above by leaving out the `@generated` in front of your
  overload.
* It might be helpful to implement matrix multiplication in a *normal* Julia
  function first.
* You can construct an expression for a sum of multiple elements like below.
```@repl lab09
Expr(:call,:+,1,2,3)
Expr(:call,:+,1,2,3) |> eval
```

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab09
@generated function Base.:*(x::StaticMatrix{T,K,M}, y::StaticMatrix{T,M,N}) where {T,K,M,N}
    zs = map(Iterators.product(1:K, 1:N) |> collect |> vec) do (k,n)
        Expr(:call, :+, [:(x[$k,$m] * y[$m,$n]) for m=1:M]...)
    end
    z = Expr(:tuple, zs...)
    :(StaticMatrix{$T,$K,$N,$(K*N)}($z))
end
nothing # hide
```
```@raw html
</p></details>
```

You can check that your matrix multiplication works by multiplying two random
matrices. Which one is faster?
```@repl lab09
a = rand(2,3)
b = rand(3,4)
c = StaticMatrix(a)
d = StaticMatrix(b)
a*b
c*d
```


## `OptionalArgChecks.jl`

The package [`OptionalArgChecks.jl`](https://github.com/simeonschaub/OptionalArgChecks.jl)
makes is possible to add checks to a function which can then be removed by
calling the function with the `@skip` macro.  For example, we can check if the
input to a function `f` is an even number
```@example lab09
function f(x::Number)
    iseven(x) || error("Input has to be an even number!")
    x
end
nothing # hide
```
If you are doing more involved argument checking it can take quite some time to
perform all your checks. However, if you want to be fast and are completely
sure that you are always passing in the correct inputs to your function, you
might want to remove them in some cases. Hence, we would like to transform the
IR of the function above
```@repl lab09
using IRTools
using IRTools: @code_ir
@code_ir f(1)
```
To some thing like this
```@repl lab09
transformed_f(x::Number) = x
@code_ir transformed_f(1)
```

### Marking Argument Checks
As a first step we will implement a macro that marks checks which we might want
to remove later by surrounding it with `:meta` expressions. This will make it
easy to detect which part of the code can be removed. A `:meta` expression can
be created like this
```@repl lab09
Expr(:meta, :mark_begin)
Expr(:meta, :mark_end)
```
and they will not be evaluated but remain in your IR. To surround an expression
with two meta expressions you can use a `:block` expression:
```@repl lab09
ex = :(x+x)
Expr(:block, :(print(x)), ex, :(print(x)))
```
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Define a macro `@mark` that takes an expression and surrounds it with two
meta expressions marking the beginning and end of a check.
**Hints**
* Defining a function `_mark(ex::Expr)` which manipulates your expressions can
  help a lot with debugging your macro.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab09
function _mark(ex::Expr)
    return Expr(
        :block,
        Expr(:meta, :mark_begin),
        esc(ex),
        Expr(:meta, :mark_end),
    )
end

macro mark(ex)
    _mark(ex)
end
nothing # hide
```
```@raw html
</p></details>
```
If you have defined a `_mark` function you can test that it works like this
```@repl lab09
_mark(:(println(x)))
```
The complete macro should work like below
```@repl lab09
function f(x::Number)
    @mark @show x
    x
end;
@code_ir f(2)
f(2)
```


### Removing Argument Checks

Now comes tricky part for which we need `IRTools.jl`.
We want to remove all lines that are between our two meta blocks.
You can delete the line that corresponds to a certain variable with the `delete!`
and the `var` functions.
E.g. deleting the line that defines variable `%4` works like this:
```@repl lab09
using IRTools: delete!, var

ir = @code_ir f(2)
delete!(ir, var(4))
```
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Write a function `skip(ir::IR)` which deletes all lines between the meta
expression `:mark_begin` and `:mark_end`.

**Hints**
You can check whether a statement is one of our meta expressions like this:
```@repl lab09
ismarkbegin(e::Expr) = Meta.isexpr(e,:meta) && e.args[1]===:mark_begin
ismarkbegin(Expr(:meta,:mark_begin))
```
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab09
ismarkend(e::Expr) = Meta.isexpr(e,:meta) && e.args[1]===:mark_end

function skip(ir)
    delete_line = false
    for (x,st) in ir
        isbegin = ismarkbegin(st.expr)
        isend   = ismarkend(st.expr)

        if isbegin
            delete_line = true
        end
        
        if delete_line
            delete!(ir,x)
        end

        if isend
            delete_line = false
        end
    end
    ir
end
nothing # hide
```
```@raw html
</p></details>
```
Your function should transform the IR of `f` like below.
```@repl lab09
ir = @code_ir f(2)
ir = skip(ir)
using IRTools: func
func(ir)(nothing, 2)  # no output from @show!
```
However, if we have a slightly more complicated IR like below this version of
our function will fail. It actually fails so badly that running
`func(ir)(nothing,2)` after `skip` will cause the build of this page to crash,
so we cannot show you the output here ;).
```@repl lab09
function g(x)
    @mark iseven(x) && println("even")
    x
end

ir = @code_ir g(2)
ir = skip(ir)
```
The crash is due to `%4` not existing anymore. We can fix this by emptying the
block in which we found the `:mark_begin` expression and branching to the
block that contains `:mark_end` (unless they are in the same block already).
If some (branching) code in between remained, it should then be removed by the
compiler because it is never reached.
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Use the functions `IRTools.block`, `IRTools.branches`, `IRTools.empty!`, and
`IRTools.branch!` to modify `skip` such that it also empties the `:mark_begin`
block, and adds a branch to the `:mark_end` block (unless they are the same
block).

**Hints**
* `block` gets you the block of IR in which a given variable is if you call e.g. `block(ir,var(4))`.
* `empty!` removes all statements in a block.
* `branches` returns all branches of a block.
* `branch!(a,b)` creates a branch from the end of block `a` to the beginning
  block `b`
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab09
using IRTools: block, branch!, empty!, branches
function skip(ir)
    delete_line = false
    orig = nothing
    for (x,st) in ir
        isbegin = ismarkbegin(st.expr)
        isend   = ismarkend(st.expr)

        if isbegin
            delete_line = true
        end

        # this part is new
        if isbegin
            orig = block(ir,x)
        elseif isend
            dest = block(ir,x)
            if orig != dest
                empty!(branches(orig))
                branch!(orig,dest)
            end
        end
        
        if delete_line
            delete!(ir,x)
        end

        if isend
            delete_line = false
        end
    end
    ir
end
nothing # hide
```
```@raw html
</p></details>
```
The result should construct valid IR for our `g` function.
```@repl lab09
g(2)
ir = @code_ir g(2)
ir = skip(ir)
func(ir)(nothing,2)
```
And it should not break when applying it to `f`.
```@repl lab09
f(2)
ir = @code_ir f(2)
ir = skip(ir)
func(ir)(nothing,2)
```

### Recursively Removing Argument Checks

The last step to finalize the `skip` function is to make it work recursively.
In the current version we can handle functions that contain `@mark` statements,
but we are not going any deeper than that. Nested functions will not be touched:
```@example lab09
foo(x) = bar(baz(x))

function bar(x)
    @mark iseven(x) && println("The input is even.")
    x
end

function baz(x)
    @mark x<0 && println("The input is negative.")
    x
end

nothing # hide
```
```@repl lab09
ir = @code_ir foo(-2)
ir = skip(ir)
func(ir)(nothing,-2)
```

For recursion we will use the macro `IRTools.@dynamo` which will make recursion
of our `skip` function a lot easier. Additionally, it will save us from all the
`func(ir)(nothing, args...)` statements. To use `@dynamo` we have to slightly
modify how we call `skip`:
```julia
@dynamo function skip(args...)
    ir = IR(args...)
    
    # same code as before that modifies `ir`
    # ...

    return ir
end

# now we can call `skip` like this
skip(f,2)
```
Now we can easily use `skip` in recursion, because we can just pass the
arguments of an expression like this:
```julia
using IRTools: xcall

for (x,st) in ir
    isexpr(st.expr,:call) || continue
    ir[x] = xcall(skip, st.expr.args...)
end
```
The function `xcall` will create an expression that calls `skip` with the given
arguments and returns `Expr(:call, skip, args...)`.  Note that you can modify
expressions of a given variable in the IR via `setindex!`.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Modify `skip` such that it uses `@dynamo` and apply it recursively to all
`:call` expressions that you ecounter while looping over the given IR.
This will dive all the way down to `Core.Builtin`s and `Core.IntrinsicFunction`s
which you cannot maniuplate anymore (because they are written in C).
You have to end the recursion at these places which can be done via multiple
dispatch of `skip` on `Builtin`s and `IntrinsicFunction`s.

Once you are done with this you can also define a macro such that you can
conveniently call `@skip` with an expression:
```julia
skip(f,2)
@skip f(2)
```
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab09
using IRTools: @dynamo, xcall, IR

# this is where we want to stop recursion
skip(f::Core.IntrinsicFunction, args...) = f(args...)
skip(f::Core.Builtin, args...) = f(args...)

@dynamo function skip(args...)
    ir = IR(args...)
    delete_line = false
    orig = nothing
    for (x,st) in ir
        isbegin = ismarkbegin(st.expr)
        isend   = ismarkend(st.expr)

        if isbegin
            delete_line = true
        end

        if isbegin
            orig = block(ir,x)
        elseif isend
            dest = block(ir,x)
            if orig != dest
                empty!(branches(orig))
                branch!(orig,dest)
            end
        end
        
        if delete_line
            delete!(ir,x)
        end

        if isend
            delete_line = false
        end

        # this part is new
        if haskey(ir,x) && Meta.isexpr(st.expr,:call)
            ir[x] = xcall(skip, st.expr.args...)
        end
    end
    return ir
end

macro skip(ex)
    ex.head == :call || error("Input expression has to be a `:call`.")
    return xcall(skip, ex.args...)
end
nothing # hide
```
```@raw html
</p></details>
```
```@repl lab09
@code_ir foo(2)
@code_ir skip(foo,2)
foo(-2)
skip(foo,-2)
@skip foo(-2)
```

## References

* [Static matrices](https://wesselb.github.io/2020/12/13/julia-learning-circle-meeting-3.html) with `@generate`d functions blog post
* [`OptionalArgChecks.jl`](https://github.com/simeonschaub/OptionalArgChecks.jl)
* IRTools [Dynamo](https://fluxml.ai/IRTools.jl/latest/dynamo/)
