using Random
using ForwardDiff
using StatsBase
using GalacticOptim
using Flux
using Optim
using Plots

Random.seed!(0)

# taken from DiffEqFlux.jl
abstract type FastLayer <: Function end

struct FastDense{F,P} <: FastLayer
    in::Int
    out::Int
    σ::F
    initial_params::P
    function FastDense(in::Integer, out::Integer, σ=identity; initW=Flux.glorot_uniform, initb=Flux.zeros32)
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

function forward_derivative(u::FastLayer,x::AbstractVector,ε::AbstractVector,θ)
    J = ForwardDiff.jacobian(i->u(i,θ),x)
    J * ε
end
function forward_derivative(u::FastLayer,xs::AbstractMatrix,ε::AbstractVector,θ)
    mapreduce(x->forward_derivative(u,x,ε,θ), hcat, eachcol(xs))
end
function forward_derivative(u::FastLayer,x::AbstractArray,v::Int,θ)
    ε = Flux.onehot(v,1:size(x,1))
    forward_derivative(u,x,ε,θ)
end
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
testloss(θ) = internal(pinn, rand(1,10), θ)

function data_loss(m::PINN, x::AbstractArray, y::AbstractArray, θ::AbstractVector)
    mean(abs2, m.u(x,θ) - y)
end

hdim = 10
m = FastChain(
    FastDense(1,hdim,tanh),
    FastDense(hdim,hdim,tanh),
    FastDense(hdim,2))
θ = initial_params(m)

lhs(u,x,θ) = forward_derivative(u,x,1,θ)
lotka_params = [0.1f0, 0.2f0, 0.3f0, 0.2f0]
rhs(u,x,θ) = lotkavolterra(u(x,θ), lotka_params)
pinn = PINN(m,lhs,rhs)

using JLD2
data = load("../../lecture_12/lotkadata.jld2")
dx = 100
de = 0.5f0
xs = Float32.(reshape(data["t"],1,:)[:,1:dx:round(Int,end*de)])
ys = Float32.(data["u"][:,1:dx:round(Int,end*de)])
#xs = Float32.(reshape([data["t"][1]],1,1))
#ys = Float32.(reshape(data["u"][:,1],:,1))

_xs = reshape(collect(0.0f0:2.0f0:100.0f0*de),1,:)
function plotprogress(θ)
    t = vec(xs)
    p1 = scatter(t, ys[1,:], label="Data x", ms=7)
    scatter!(p1, t, ys[2,:], label="Data y", ms=7)
    t = data["t"]
    plot!(p1, t, data["u"][1,:], label=false, lw=3, c=:gray)
    plot!(p1, t, data["u"][2,:], label=false, lw=3, c=:gray)

    us = pinn.u(reshape(t,1,:),θ)
    plot!(p1, t, us[1,:], label="PINN x", lw=3, c=1)
    plot!(p1, t, us[2,:], label="PINN y", lw=3, c=2)
    plot!(p1, xlim=(_xs[1],_xs[end]))
end

function loss(θ,p=nothing)
    l1 = internal_loss(pinn, _xs, θ)
    l2 = data_loss(pinn, xs, ys, θ)
    l1+l2, (l1,l2)
end

function callback(θ,l,ls)
    (internal, dataloss) = ls
    plotprogress(θ) |> display
    @info l internal dataloss
    false
end

display(loss(θ))
ForwardDiff.gradient(first ∘ loss, θ)

func = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(func, θ)
opt  = LBFGS()

@time res = solve(prob, opt, maxiters=10_000, cb=Flux.throttle(callback,1), progress=true)
p = plotprogress(res.minimizer)
