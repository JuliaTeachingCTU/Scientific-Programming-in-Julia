using LinearAlgebra
using StatsBase
using Distributions

include("gaussnum.jl")
include("ode-solver.jl")

# Define & Solve ODE

function lotkavolterra(x,θ)
    α, β, γ, δ = θ
    x₁, x₂ = x

    dx₁ = α*x₁ - β*x₁*x₂
    dx₂ = δ*x₁*x₂ - γ*x₂

    [dx₁, dx₂]
end

struct GaussODEProblem{F,T<:Tuple{Number,Number},U<:AbstractMatrix,P<:AbstractVector,I} <: AbstractODEProblem
    f::F
    tspan::T
    u0::U
    θ::P
    θ_idx::I
    size::Int
end
Base.size(prob::GaussODEProblem) = size(prob.u0,1)
statesize(prob::GaussODEProblem) = prob.size

function GaussODEProblem(f,tspan,u0::V,θ::V) where V<:AbstractVector{<:GaussNum}
    θ_idx = findall(isuncertain, θ)

    μ = vcat(mean.(u0), mean.(θ[θ_idx]))
    d = length(μ)
    Σ = vcat(var.(u0), var.(θ[θ_idx])) |> Diagonal |> Matrix
    Σ½ = cholesky(Σ).L
    Qp = sqrt(d)*[I(d) -I(d)]
    Xp = μ .+ Σ½*Qp

    GaussODEProblem(f,tspan,Xp,mean.(θ),θ_idx,length(u0))
end

function setmean!(prob::GaussODEProblem,xi)
    prob.θ[prob.θ_idx] .= xi[(statesize(prob)+1):end]
    xi[1:statesize(prob)]
end

struct GaussODESolver{S<:ODESolver} <: ODESolver
    solver::S
end

function (s::GaussODESolver)(prob::GaussODEProblem, Xp, t)
    Xp = reduce(hcat, map(eachcol(Xp)) do xi
        ui = setmean!(prob, xi)
        ui = s.solver(prob, ui, t)[1]
        xi[1:statesize(prob)] .= ui
        xi
    end)
    Xp, t+s.solver.dt
end

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


gus = map(us) do u
    u = u[1:statesize(prob),:]
    μ = mean(u,dims=2) |> vec
    σ = std(u,dims=2) |> vec
    GaussNum.(μ,σ)
end
gus = reduce(hcat,gus)
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
    us = reduce(hcat,us)
    push!(Us,us)
    #plot!(p2, t, us[1,:], c=1, alpha=0.5, label=false)
    #plot!(p2, t, us[2,:], c=2, alpha=0.5, label=false)
end

plot!(p2, t, GaussNum.(mean(Us)[1,:],std(Us)[1,:]), lw=3)
plot!(p2, t, GaussNum.(mean(Us)[2,:],std(Us)[2,:]), lw=3)
display(plot(p1,p2))
