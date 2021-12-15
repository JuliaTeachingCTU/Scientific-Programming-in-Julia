using Random
using ForwardDiff
using StatsBase
using GalacticOptim
using Flux
using Optim
using Plots

Random.seed!(0)

# taken from Flux.jl
nfan() = 1, 1 # fan_in, fan_out
nfan(n) = 1, n # A vector is treated as a n×1 matrix
nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
nfan(dims::Tuple) = nfan(dims...)
glorot_uniform(rng::AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(nfan(dims...)))
glorot_uniform(dims...) = glorot_uniform(Random.GLOBAL_RNG, dims...)
glorot_uniform(rng::AbstractRNG) = (dims...) -> glorot_uniform(rng, dims...)

# taken from DiffEqFlux.jl
abstract type FastLayer <: Function end
struct FastDense{F,P} <: FastLayer
    in::Int
    out::Int
    σ::F
    initial_params::P
    function FastDense(in::Integer, out::Integer, σ=identity; initW=glorot_uniform, initb=zeros)
        initial_params() = vcat(vec(initW(out,in)), initb(out))
        new{typeof(σ),typeof(initial_params)}(in,out,σ,initial_params)
    end
end

initial_params(f::FastDense) = f.initial_params()
paramlength(f::FastDense) = f.out*(f.in+1)

function (f::FastDense)(x,θ)
    W = reshape(θ[1:(f.out*f.in)], f.out, f.in)
    b = θ[(f.out*f.in+1):end]
    f.σ.(W*x .+ b)
end

struct FastChain{T<:Tuple} <: FastLayer
    layers::T
end
FastChain(ls...) = FastChain(ls)

applychain(::Tuple{}, x, p) = x
function applychain(fs::Tuple, x, p)
    applychain(Base.tail(fs), first(fs)(x,p[1:paramlength(first(fs))]), p[(paramlength(first(fs))+1):end])
end
(c::FastChain)(x,p) = applychain(c.layers, x, p)
paramlength(c::FastChain) = sum(paramlength(x) for x in c.layers)
initial_params(c::FastChain) = vcat(initial_params.(c.layers)...)


##########  PINN  ##############################################################

function lotkavolterra(u::AbstractVector,θ)
    α, β, γ, δ = θ
    u₁, u₂ = u

    du₁ = α*u₁ - β*u₁*u₂
    du₂ = δ*u₁*u₂ - γ*u₂

    [du₁, du₂]
end
#lotkavolterra(u::AbstractMatrix,θ) = mapreduce(ui->lotkavolterra(ui,θ), hcat, eachcol(u))
function lotkavolterra(u::AbstractMatrix,θ)
    α, β, γ, δ = θ
    u₁, u₂ = u[1,:], u[2,:]

    du₁ = α.*u₁ - β.*u₁.*u₂
    du₂ = δ.*u₁.*u₂ - γ.*u₂

    [du₁ du₂]'
end

function forward_derivative(u,x::AbstractVector,ε::Vector,θ::Vector)
    J(x) = ForwardDiff.jacobian(i->u(i,θ),x)
    J(x) * ε
end
function forward_derivative(u,xs::AbstractMatrix,ε::Vector,θ::Vector)
    gs = map(eachcol(xs)) do x
        forward_derivative(u,x,ε,θ)
    end
    reduce(hcat, gs)
end
function get_v(dim, der_num)
    map(1:dim) do i
        i==der_num ? 1 : 0
    end
end
forward_derivative(u,x::AbstractArray,v::Int,θ::Vector) = forward_derivative(u,x,get_v(size(x,1),v),θ)
function get_ε(dim, der_num)
    epsilon = cbrt(eps(Float32))
    e = map(1:dim) do i
        # NOTE: needed to prevent Zygote mutation error
        i==der_num ? epsilon : 0f0
    end
end
function numeric_derivative(u::FastLayer,x,ε::Vector,order::Int,θ::Vector)
    _epsilon = 1 / (2*cbrt(eps(eltype(θ))))
    if order > 1
        a = numeric_derivative(u,x .+ ε, ε, order-1, θ)
        b = numeric_derivative(u,x .- ε, ε, order-1, θ)
        (a .- b) .* _epsilon
    else
        (u(x .+ ε,θ) .- u(x .- ε,θ)) .* _epsilon
    end
end


struct PINN{U<:FastLayer,L,R}
    u::U
    lhs::L
    rhs::R
end

function internal_loss(m::PINN, x::AbstractArray, θ::AbstractVector)
    u = m.u
    mean(abs2, m.lhs(u,x,θ) - m.rhs(u,x,θ))
end

function data_loss(m::PINN, x::AbstractArray, y::AbstractArray, θ::AbstractVector)
    mean(abs2, m.u(x,θ) - y)
end

lhs(u,x,θ) = forward_derivative(u,x,1,θ)
#lhs(u,x,θ) = numeric_derivative(u,x,get_ε(1,1),1,θ)
function rhs(u,x,θ)
    p = [0.1, 0.2, 0.3, 0.2]
    lotkavolterra(u(x,θ),p)
end

hdim = 10
m = FastChain(
    FastDense(1,hdim,tanh),
    FastDense(hdim,hdim,tanh),
    FastDense(hdim,2))
θ = initial_params(m)
pinn = PINN(m,lhs,rhs)
using JLD2
data = load("../../lecture_12/lotkadata.jld2")
dx = 100
de = 0.5
xs = reshape(data["t"],1,:)[:,1:dx:round(Int,end*de)]
ys = data["u"][:,1:dx:round(Int,end*de)]

_xs = reshape(collect(0.0:2.0:100.0*de),1,:)
function plotprogress(pinn,θ,xs,ys)
    t = vec(xs)
    p1 = scatter(t, ys[1,:], label="x")
    scatter!(p1, t, ys[2,:], label="x")
    plot!(p1, data["t"], data["u"][1,:], label=false, lw=3, c=:gray)
    plot!(p1, data["t"], data["u"][2,:], label=false, lw=3, c=:gray)

    us = pinn.u(_xs,θ)
    plot!(p1, vec(_xs), us[1,:], label="fit x", lw=3)
    plot!(p1, vec(_xs), us[2,:], label="fit y", lw=3)
    plot!(p1, xlim=(_xs[1],_xs[end]))
end

function loss(θ,p=nothing)
    l1 = internal_loss(pinn, _xs, θ) * 1e0
    l2 = data_loss(pinn, xs, ys, θ)
    l1+l2, (l1,l2)
end

function callback(θ,l,ls)
    (internal, dataloss) = ls
    plotprogress(pinn,θ,xs,ys) |> display
    @info l internal dataloss
    false
end

display(loss(θ))
ForwardDiff.gradient(first ∘ loss, θ)

func = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(func, θ)
opt  = LBFGS()

@time solve(prob, opt, maxiters=4000, cb=Flux.throttle(callback,1), progress=true)
