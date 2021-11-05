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

function loss(network,x::TrackedArray,y::TrackedArray)
    err = network(x) - y
    err |> abs2 |> sum
end

# TODO: can we make broadcasting work? so that we can use batches?
x = track(rand(2))
y = track(rand(2))
l = loss(network,x,y)
l.grad = 1.0
accum!(network.layers[1].W)

#using FiniteDifferences
#df = grad(central_fdm(5,1), n->loss(n,x,y), network)[1]
