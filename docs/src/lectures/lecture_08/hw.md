# [Homework 08](@id hw08)

In this homework you will write an additional rule for our scalar reverse AD
from the lab. For this homework, please write all your code in one file `hw.jl`
which you have to zip and upload to BRUTE as usual. The solution to the lab is
below.
```@example hw08
mutable struct TrackedReal{T<:Real}
    data::T
    grad::Union{Nothing,T}
    children::Dict
    # this field is only need for printing the graph. you can safely remove it.
    name::String
end

track(x::Real,name="") = TrackedReal(x,nothing,Dict(),name)

function Base.show(io::IO, x::TrackedReal)
    t = isempty(x.name) ? "(tracked)" : "(tracked $(x.name))"
    print(io, "$(x.data) $t")
end

function accum!(x::TrackedReal)
    if isnothing(x.grad)
        x.grad = sum(w*accum!(v) for (v,w) in x.children)
    end
    x.grad
end

function Base.:*(a::TrackedReal, b::TrackedReal)
    z = track(a.data * b.data, "*")
    a.children[z] = b.data  # dz/da=b
    b.children[z] = a.data  # dz/db=a
    z
end

function Base.:+(a::TrackedReal{T}, b::TrackedReal{T}) where T
    z = track(a.data + b.data, "+")
    a.children[z] = one(T)
    b.children[z] = one(T)
    z
end

function Base.sin(x::TrackedReal)
    z = track(sin(x.data), "sin")
    x.children[z] = cos(x.data)
    z
end

function gradient(f, args::Real...)
    ts = track.(args)
    y  = f(ts...)
    y.grad = 1.0
    accum!.(ts)
end
```

We will use it to compute the derivative of the Babylonian square root.
```@example hw08
babysqrt(x, t=(1+x)/2, n=10) = n==0 ? t : babysqrt(x, (t+x/t)/2, n-1)
nothing # hide
```
In order to differentiate through `babysqrt` you will need a reverse rule for `/`
for `Base.:/(TrackedReal,TrackedReal)` as well as the cases where you divide with
constants in volved (e.g. `Base.:/(TrackedReal,Real)`).

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (2 points)</header>
<div class="admonition-body">
```
Write the reverse rules for `/`  and the missing rules for `+` such that you can
differentiate through division and addition with and without constants.
```@raw html
</div></div>
```
```@setup hw08
function Base.:/(a::TrackedReal, b::TrackedReal)
    z = track(a.data / b.data)
    a.children[z] = 1/b.data
    b.children[z] = -a.data / b.data^2
    z
end
function Base.:/(a::TrackedReal, b::Real)
    z = track(a.data/b)
    a.children[z] = 1/b
    z
end

function Base.:+(a::Real, b::TrackedReal{T}) where T
    z = track(a + b.data, "+")
    b.children[z] = one(T)
    z
end
Base.:+(a::TrackedReal,b::Real) = b+a
```

You can verify your solution with the `gradient` function.
```@repl hw08
gradient(babysqrt, 2.0)
1/(2babysqrt(2.0))
```
