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
(m::Dense)(x) = m.f(m.W*x + m.b)


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
f(xs) = [f(xs[1],xs[2])]

# data
xs = map(1:30) do i
    x = rand(-4:0.1:4)
    y = rand(-2:0.1:2)
    [x,y]
end
ys = track.(f.(xs))
xs = track.(xs)


hdim = 15
network = Chain(
    Dense(2,hdim,σ),
    Dense(hdim,hdim,σ),
    Dense(hdim,1)
)
function loss(network,xs,ys)
    errs = hcat([abs2(network(x)-y) for (x,y) in zip(xs,ys)]...)
    sum(errs)
end

using Plots
color_scheme = cgrad(:RdYlBu_5, rev=true)

forward(network,x,y) = network(track([x,y])).data[1]

# training
anim = @animate for i in 1:2000
    ReverseDiff.reset!.(xs)
    ReverseDiff.reset!.(ys)
    ReverseDiff.reset!(network)
    l = train_step!(loss, network, xs, ys, λ=0.003)
    if mod1(i,50) == 1
        @info i l
        p1 = contour(-4:0.3:4, -2:0.3:2, f, fill=true, c=color_scheme, xlabel="x", ylabel="y", title="Truth")
        p2 = contour(-4:0.3:4, -2:0.3:2, (x,y)->forward(network,x,y), fill=true, c=color_scheme, xlabel="x",title="Iteration: $i")
        p = plot(p1,p2,size=(1200,400)) |> display
    end
end every 50

gif(anim, "anim.gif", fps=15)
error()

for i in 1:20000
    ReverseDiff.reset!.(xs)
    ReverseDiff.reset!.(ys)
    ReverseDiff.reset!(network)
    l = train_step!(loss, network, xs, ys, λ=0.002)
    if mod1(i,1000) == 1
        @info i l
        p1 = contour(-4:0.3:4, -2:0.3:2, f, fill=true, c=color_scheme, xlabel="x", ylabel="y", title="Truth")
        p2 = contour(-4:0.3:4, -2:0.3:2, (x,y)->forward(network,x,y), fill=true, c=color_scheme, xlabel="x",title="Iteration: $i")
        plot(p1,p2) |> display
    end
end
