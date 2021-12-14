using LinearAlgebra
using StatsBase
using Distributions

abstract type AbstractODEProblem end

struct ODEProblem{F,T<:Tuple{Number,Number},U<:AbstractVector,P<:AbstractVector} <: AbstractODEProblem
    f::F
    tspan::T
    u0::U
    θ::P
end


abstract type ODESolver end

struct Euler{T} <: ODESolver
    dt::T
end

function (solver::Euler)(prob::AbstractODEProblem, u, t)
    f, θ, dt  = prob.f, prob.θ, solver.dt
    (u + dt*f(u,θ), t+dt)
end
struct RK2{T} <: ODESolver
    dt::T
end
function (solver::RK2)(prob::AbstractODEProblem, u, t)
    f, θ, dt  = prob.f, prob.θ, solver.dt
    du = f(u,θ)
    uh = u + du*dt
    u + dt/2*(du + f(uh,θ)), t+dt
end


function solve(prob::AbstractODEProblem, solver::ODESolver)
    t = prob.tspan[1]; u = prob.u0
    us = [u]; ts = [t]
    while t < prob.tspan[2]
        (u,t) = solver(prob, u, t)
        push!(us,u)
        push!(ts,t)
    end
    ts, reduce(hcat,us)
end


# Define & Solve ODE

function lotkavolterra(x,θ)
    α, β, γ, δ = θ
    x₁, x₂ = x

    dx₁ = α*x₁ - β*x₁*x₂
    dx₂ = δ*x₁*x₂ - γ*x₂

    [dx₁, dx₂]
end

struct GaussODEProblem{F,T<:Tuple{Number,Number},U<:AbstractVector,
                       P<:AbstractVector,D<:MvNormal} <: AbstractODEProblem
    f::F
    tspan::T
    u0::U
    θ::P
    g::D
    u_idx
    θ_idx
end
Base.size(prob::GaussODEProblem) = length(prob.u_idx) + length(prob.θ_idx)

function GaussODEProblem(f,tspan,u0,θ)
    u_idx = findall(isuncertain, u0)
    θ_idx = findall(isuncertain, θ)
    μ = vcat(mean.(u0[u_idx]), mean.(θ[θ_idx]))
    Σ = vcat(var.(u0[u_idx]), var.(θ[θ_idx])) |> Diagonal |> Matrix
    g = MvNormal(μ,Σ)
    GaussODEProblem(f,tspan,mean.(u0),mean.(θ),g,u_idx,θ_idx)
end

function setmean!(prob::GaussODEProblem,xi)
    prob.θ[prob.θ_idx] .= xi[(length(prob.u_idx)+1):end]
    xi[1:length(prob.u_idx)]
end

struct GaussODESolver{S<:ODESolver} <: ODESolver
    solver::S
end

function (s::GaussODESolver)(prob::GaussODEProblem, u::MvNormal, t, Xp)
    u_idx, θ_idx = prob.u_idx, prob.θ_idx
    d  = size(prob)
    #μ  = mean(u)
    #Σ  = cov(u)
    #Σ½ = cholesky(Σ).L
    #Qp = sqrt(d)*[I(d) -I(d)]
    #Xp = μ .+ Σ½*Qp

    for i in 1:size(Xp,2)
        xi = @view Xp[:,i]
        ui = setmean!(prob, xi)
        ui = s.solver(prob, ui, t)[1]
        xi[1:length(u_idx)] .= ui[u_idx]
    end

    μ = mean(Xp,dims=2) |> vec
    Σ = (Xp .- μ)*(Xp .- μ)' / (2d)
    #Σ = cov(Xp,dims=2)

    MvNormal(μ,Σ), t+s.solver.dt
end

function solve(prob::GaussODEProblem, solver::GaussODESolver)
    t = prob.tspan[1]; u = prob.g
    us = [u]; ts = [t]

    d  = size(prob)
    μ  = mean(u)
    Σ  = cov(u)
    Σ½ = cholesky(Σ).L
    Qp = sqrt(d)*[I(d) -I(d)]
    Xp = μ .+ Σ½*Qp

    while t < prob.tspan[2]
        (u,t) = solver(prob, u, t, Xp)
        push!(us,u)
        push!(ts,t)
    end
    ts, us
end

struct GaussNum{T<:Real} <: Real
    μ::T
    σ::T
end
StatsBase.mean(x::GaussNum) = x.μ
StatsBase.var(x::GaussNum) = x.σ^2
StatsBase.std(x::GaussNum) = x.σ
GaussNum(x,y) = GaussNum(promote(x,y)...)
±(x,y) = GaussNum(x,y)
Base.convert(::Type{T}, x::T) where T<:GaussNum = x
Base.convert(::Type{GaussNum{T}}, x::Number) where T = GaussNum(x,zero(T))
Base.promote_rule(::Type{GaussNum{T}}, ::Type{S}) where {T,S} = GaussNum{T}
Base.promote_rule(::Type{GaussNum{T}}, ::Type{GaussNum{T}}) where T = GaussNum{T}

gaussnums(x::MvNormal) = GaussNum.(mean(x), sqrt.(var(x)))
gaussnums(xs::Vector{<:MvNormal}) = reduce(hcat, gaussnums.(xs))
isuncertain(x::GaussNum) = x.σ!=0
isuncertain(x::Number) = false


function lotkavolterra(x,θ)
    α, β, γ, δ = θ
    x₁, x₂ = x

    dx₁ = α*x₁ - β*x₁*x₂
    dx₂ = δ*x₁*x₂ - γ*x₂

    [dx₁, dx₂]
end

θ = [0.1±0.01, 0.2, 0.3, 0.2]
u0 = [1.0±0.1, 1.0±0.1]
tspan = (0., 100.)
prob = GaussODEProblem(lotkavolterra,tspan,u0,θ)
solver = GaussODESolver(RK2(0.1))
t, us = solve(prob,solver)


using Plots
@recipe function plot(ts::AbstractVector, xs::AbstractVector{<:GaussNum})
    # you can set a default value for an attribute with `-->`
    # and force an argument with `:=`
    μs = [x.μ for x in xs]
    σs = [x.σ for x in xs]
    @series begin
        :seriestype := :path
        # ignore series in legend and color cycling
        primary := false
        linecolor := nothing
        fillcolor := :gray
        fillalpha := 0.5
        fillrange := μs .- σs
        # ensure no markers are shown for the error band
        markershape := :none
        # return series data
        ts, μs .+ σs
    end
    ts, μs
end

gus = gaussnums(us)
p1 = plot(t, gus[1,:], lw=3)
plot!(p1, t, gus[2,:], lw=3) |> display


solver = RK2(0.1)
prob = ODEProblem(lotkavolterra,tspan,u0,θ)

function Base.rand(x::AbstractVector{<:GaussNum{T}}) where T
    mean.(x) .+ std.(x) .* randn(T,length(x))
end
Base.rand(prob::ODEProblem) = ODEProblem(prob.f, prob.tspan, rand(prob.u0), rand(prob.θ))

p2 = plot()
Us = []
Ss = []
for _ in 1:200
    t, us = solve(rand(prob),solver)
    push!(Us,us)
    #plot!(p2, t, us[1,:], c=1, alpha=0.5, label=false)
    #plot!(p2, t, us[2,:], c=2, alpha=0.5, label=false)
end

plot!(p2, t, GaussNum.(mean(Us)[1,:],std(Us)[1,:]), lw=3)
plot!(p2, t, GaussNum.(mean(Us)[2,:],std(Us)[2,:]), lw=3)
display(plot(p1,p2))
