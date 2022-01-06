using Scientific_Programming_in_Julia
using Base: tail

import Scientific_Programming_in_Julia.ReverseDiff


##########  Broadcasting  ######################################################

import Base.Broadcast: broadcasted

function broadcasted(::typeof(+), A::TrackedMatrix, B::TrackedVector)
    C = track(A.data .+ B.data)
    A.children[C] = Δ -> Δ
    B.children[C] = Δ -> sum(Δ, dims = 2)[:]
    C
end
function broadcasted(::typeof(identity), A::TrackedArray)
    C = track(A.data)
    A.children[C] = Δ->Δ
    C
end

function broadcasted(::typeof(σ), x::TrackedArray)
    z = track(σ.(x.data))
    d = z.data
    x.children[z] = Δ -> Δ .* d .* (1 .- d)
    z
end
function broadcasted(::typeof(+), X::TrackedArray, Y::TrackedArray)
    Z = track(X.data + Y.data)
    X.children[Z] = Δ -> Δ
    Y.children[Z] = Δ -> Δ
    Z
end
function Base.:+(X::TrackedReal, Y::TrackedReal)
    Z = track(X.data + Y.data)
    X.children[Z] = Δ -> Δ
    Y.children[Z] = Δ -> Δ
    Z
end

# relu(x::Real) = max(0,x)
# function relu(x::TrackedReal)
#     y = track(relu(x.data))
#     x.children[y] = Δ -> Δ > 0
#     y
# end
# function broadcasted(::typeof(relu), A::TrackedArray)
#     C = track(relu.(A.data))
#     A.children[C] = Δ -> Δ .> 0
#     C
# end



##########  Network definition  ################################################
using Base: tail
struct Dense{M<:TrackedMatrix, V<:TrackedVector, F}
    W::M
    b::V
    f::F
end

Dense(W::Matrix, b::Vector, f) = Dense(track(W),track(b),f)
Dense(in,out,f=identity) = Dense(rand(out,in) .-0.5, rand(out) .-0.5, f)

(m::Dense)(x) = m.f.(m.W*x .+ m.b)

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

loss(network,xs,ys) = sum(abs2(network(xs) - ys))

using Plots
color_scheme = cgrad(:RdYlBu_5, rev=true)

# only for plotting
forward(network,x,y) = _y = network(track([x,y])).data

onecold(x::AbstractMatrix) = [y.I[1] for y in argmax(x,dims=1)]
onecold(x::AbstractVector) = argmax(x)

for i in 1:500
    ReverseDiff.reset!(xs)
    ReverseDiff.reset!(ys)
    ReverseDiff.reset!(network)
    l = train_step!(loss, network, xs, ys, λ=0.00005)
    if mod1(i,30) == 1
        @info i l

        p1 = contour(-5:0.3:5, -5:0.2:5, (x,y)->onecold(forward(network,x,y)), fill=true, c=color_scheme)
        scatter!(p1, xs[1,:].data, xs[2,:].data, c=onecold(ys.data)|>vec, xlims=(-5,5), ylims=(-5,5))
        display(p1)
    end
end


#anim = @animate for i in 1:2000
#    ReverseDiff.reset!(xs)
#    ReverseDiff.reset!(ys)
#    ReverseDiff.reset!(network)
#    l = train_step!(loss, network, xs, ys, λ=0.01)
#    if mod1(i,50) == 1
#        @info i l
#        ȳs = argmax(network(xs).data, dims=1)
#        ȳs = [y.I[1] for y in ȳs]
#        p2 = scatter(xs[1,:].data, xs[2,:].data, c=vec(ȳs), xlims=(-5,5), ylims=(-5,5))
#        display(p2)
#    end
#end every 50
#
#gif(anim, "anim.gif", fps=15)
