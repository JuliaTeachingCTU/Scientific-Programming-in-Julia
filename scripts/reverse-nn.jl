using Scientific_Programming_in_Julia
using Base: tail

import Scientific_Programming_in_Julia.ReverseDiff

struct Dense{M<:TrackedMatrix, V<:TrackedVector, F}
    W::M
    b::V
    f::F
end

Dense(W::Matrix, b::Vector, f) = Dense(track(W),track(b),f)
Dense(in,out,f=identity) = Dense(rand(out,in), rand(out), f)
(m::Dense)(x) = m.f(m.W*x + m.b)


struct Chain{T<:Tuple}
    layers::T
end
Chain(ls...) = Chain(ls)

(m::Chain)(x) = applychain(m.layers, x)
applychain(ls::Tuple{}, x) = x
applychain(ls::Tuple, x) = applychain(tail(ls), first(ls)(x))


function ReverseDiff.reset!(m::Chain)
    map(ReverseDiff.reset!, m.layers)
    nothing
end

function ReverseDiff.reset!(m::Dense)
    ReverseDiff.reset!(m.W)
    ReverseDiff.reset!(m.b)
    nothing
end

network = Chain(
    Dense(2,4,σ),
    Dense(4,3,σ),
    Dense(3,2)
)

function loss(network,xs,ys)
    errs = hcat([abs2(network(x)-y) for (x,y) in zip(xs,ys)]...)
    sum(errs)
end

xs = [track(rand(2)) for _ in 1:10]
ys = [track(rand(2)) for _ in 1:10]
l  = loss(network,xs,ys)

W = network.layers[1].W
l.grad = 1.0
accum!(W)

display(W.grad)
