using Scientific_Programming_in_Julia
using Base: tail

import Scientific_Programming_in_Julia.ReverseDiff

using Base: tail
struct Dense{M<:TrackedMatrix, V<:TrackedVector, F}
    W::M
    b::V
    f::F
end


Dense(W::Matrix, b::Vector, f) = Dense(track(W),track(b),f)
Dense(in,out,f=identity) = Dense(rand(out,in) .-0.5, rand(out) .-0.5, f)


import Base.Broadcast: broadcasted

function broadcasted(::typeof(+), A::TrackedMatrix, B::TrackedVector)
    C = track(A.data .+ B.data)
    A.children[C] = Δ -> Δ
    B.children[C] = Δ -> sum(Δ, dims = 2)[:]
    C
end

function (m::Dense)(x::AbstractMatrix)
    m.f(m.W*x .+ m.b)
end
function (m::Dense)(x::AbstractVector)
    m.f(m.W*x + m.b)
end



struct Chain{T<:Tuple}
    layers::T
end
Chain(ls...) = Chain(ls)

(m::Chain)(x) = applychain(m.layers, x)
applychain(ls::Tuple{}, x) = x
applychain(ls::Tuple, x) = applychain(tail(ls), first(ls)(x))

params(m::Dense) = (W=m.W, b=m.b)
params(m::Chain) = params(m.layers)
params(ls::Tuple) = (params(first(ls))..., params(tail(ls))...)
params(ls::Tuple{}) = ls

function ReverseDiff.reset!(m::Chain)
    map(ReverseDiff.reset!, m.layers)
    nothing
end

function ReverseDiff.reset!(m::Dense)
    ReverseDiff.reset!(m.W)
    ReverseDiff.reset!(m.b)
    nothing
end

function train_step!(loss, network, xs, ys; λ=0.01)
    l = loss(network, xs, ys)
    l.grad = 1.0
    for w in params(network)
        w.data .-= λ .* accum!(w)
    end
    l
end


# task
f(x,y) = y^2 + sin(x)
f(xs::AbstractVector) = [f(xs[1],xs[2])]
f(xs::AbstractMatrix) = reshape(f.(xs[1,:],xs[2,:]), 1, :)

using Random
Random.seed!(0)

## data
#xs = map(1:30) do i
#    x = rand(-4:0.1:4)
#    y = rand(-2:0.1:2)
#    [x,y]
#end
#xs = reduce(hcat, xs)
#ys = track(f(xs))
#xs = track(xs)

function flower(n; npetals = 8)
    n = div(n, npetals)
    x = mapreduce(hcat, (1:npetals) .* (2π/npetals)) do θ
        ct = cos(θ)
        st = sin(θ)

        x0 = tanh.(randn(1, n) .- 1) .+ 4.0 .+ 0.05.* randn(1, n)
        y0 = randn(1, n) .* 0.3

        x₁ = x0 * cos(θ) .- y0 * sin(θ)
        x₂ = x0 * sin(θ) .+ y0 * cos(θ)
        vcat(x₁, x₂)
    end
    _y = mapreduce(i -> fill(i, n), vcat, 1:npetals)
    display(_y')
    y = zeros(npetals, length(_y))
    for i in 1:length(_y)
        y[_y[i],i] = 1
    end

    #y = foreach(i -> y[_y[i], i] = 1, _y)
    #display(y)
    #error()
    Float32.(x), Float32.(y)
end
(xs,ys) = track.(flower(900))


hdim = 15
network = Chain(
    Dense(2,hdim,σ),
    Dense(hdim,hdim,σ),
    Dense(hdim,size(ys,1))
)
function loss(network,xs,ys)
    sum(abs2(network(xs) - ys))
    # errs = hcat([abs2(network(x)-y) for (x,y) in zip(xs,ys)]...)
    # sum(errs)
end

# l = loss(network,xs,ys)
# l.grad = 1.0
# display(l)
# display(accum!(network.layers[1].W))
# 
# error()

using Plots
color_scheme = cgrad(:RdYlBu_5, rev=true)

forward(network,x,y) = network(track([x,y])).data[1]

# training
#anim = @animate for i in 1:2000
#    ReverseDiff.reset!(xs)
#    ReverseDiff.reset!(ys)
#    ReverseDiff.reset!(network)
#    l = train_step!(loss, network, xs, ys, λ=0.01)
#    if mod1(i,50) == 1
#        @info i l
#        p1 = contour(-4:0.3:4, -2:0.3:2, f, fill=true, c=color_scheme, xlabel="x", ylabel="y", title="Truth")
#        p2 = contour(-4:0.3:4, -2:0.3:2, (x,y)->forward(network,x,y), fill=true, c=color_scheme, xlabel="x",title="Iteration: $i")
#        p = plot(p1,p2,size=(1200,400)) |> display
#    end
#end every 50
#
#gif(anim, "anim.gif", fps=15)

for i in 1:2000
    ReverseDiff.reset!(xs)
    ReverseDiff.reset!(ys)
    ReverseDiff.reset!(network)
    l = train_step!(loss, network, xs, ys, λ=0.0001)
    if mod1(i,50) == 1
        @info i l
        ȳs = argmax(network(xs).data, dims=1)
        ȳs = [y.I[1] for y in ȳs]
        display(network(xs))
        display(ys)
        display(ȳs)
        p = scatter(xs[1,:].data, xs[2,:].data, c=vec(ȳs))
        display(p)
        # p1 = contour(-4:0.3:4, -2:0.3:2, f, fill=true, c=color_scheme, xlabel="x", ylabel="y", title="Truth")
        # p2 = contour(-4:0.3:4, -2:0.3:2, (x,y)->forward(network,x,y), fill=true, c=color_scheme, xlabel="x",title="Iteration: $i")
        # p = plot(p1,p2,size=(1200,400)) |> display
    end
end
