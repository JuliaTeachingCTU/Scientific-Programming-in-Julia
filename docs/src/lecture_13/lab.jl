using LinearAlgebra
using Statistics
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

function (solver::Euler)(prob::ODEProblem, u, t)
    f, θ, dt  = prob.f, prob.θ, solver.dt
    (u + dt*f(u,θ), t+dt)
end
struct RK2{T} <: ODESolver
    dt::T
end
function (solver::RK2)(prob::ODEProblem, u, t)
    f, θ, dt  = prob.f, prob.θ, solver.dt
    du = f(u,θ)
    uh = u + du*dt
    u + dt/2*(du + f(uh,θ)), t+dt
end


function solve(prob::ODEProblem, solver::ODESolver)
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

struct GaussODEProblem{P<:ODEProblem,D<:MvNormal} <: AbstractODEProblem
    mean::P
    u0::D
end
Base.size(prob::GaussODEProblem) = length(prob.mean.u0)


struct GaussODESolver{S<:ODESolver} <: ODESolver
    solver::S
end

setmean!(prob,xi) = xi

function (s::GaussODESolver)(prob::GaussODEProblem, u::MvNormal, t)
    d  = size(prob)
    μ  = mean(u)
    Σ  = cov(u)
    Σ½ = cholesky(Σ).L
    Qp = sqrt(d)*[I(d) -I(d)]
    Xp = μ .+ Σ½*Qp

    for i in 1:2d
        xi = @view Xp[:,i]
        ui = setmean!(prob, xi)
        xi .= s.solver(prob.mean, ui, t)[1]
    end

    μ = mean(Xp,dims=2) |> vec
    Σ = (Xp .- μ)*(Xp .- μ)' / (2d)

    MvNormal(μ,Σ), t+s.solver.dt
end

function solve(prob::GaussODEProblem, solver::GaussODESolver)
    t = prob.mean.tspan[1]; u = prob.u0
    us = [u]; ts = [t]
    while t < prob.mean.tspan[2]
        (u,t) = solver(prob, u, t)
        push!(us,u)
        push!(ts,t)
    end
    ts, us
end


function lotkavolterra(x,θ)
    α, β, γ, δ = θ
    x₁, x₂ = x

    dx₁ = α*x₁ - β*x₁*x₂
    dx₂ = δ*x₁*x₂ - γ*x₂

    [dx₁, dx₂]
end

θ = [0.1,0.2,0.3,0.2]
u0 = [1.0,1.0]
tspan = (0.,100.)
oprob = ODEProblem(lotkavolterra,tspan,u0,θ)

d = length(u0)
Σ = 0.1*I(d) |> Matrix
u = MvNormal(u0,Σ)
gprob = GaussODEProblem(oprob,u)

solver = GaussODESolver(Euler(0.1))
#solver(gprob, u, 0.0) |> display

t, us = solve(gprob,solver)



struct GaussNum{T<:Real} <: Real
    μ::T
    σ::T
end
mu(x::GaussNum) = x.μ
sig(x::GaussNum) = x.σ
GaussNum(x,y) = GaussNum(promote(x,y)...)
±(x,y) = GaussNum(x,y)
Base.convert(::Type{T}, x::T) where T<:GaussNum = x
Base.convert(::Type{GaussNum{T}}, x::Number) where T = GaussNum(x,zero(T))
Base.promote_rule(::Type{GaussNum{T}}, ::Type{S}) where {T,S} = GaussNum{T}
Base.promote_rule(::Type{GaussNum{T}}, ::Type{GaussNum{T}}) where T = GaussNum{T}

gaussnums(x::MvNormal) = GaussNum.(mean(x), sqrt.(var(u)))
gaussnums(xs::Vector{<:MvNormal}) = reduce(hcat, gaussnums.(xs))

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
        fillcolor := :lightgray
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
plot(t, gus[1,:], lw=3)
plot!(t, gus[2,:], lw=3) |> display




error()







θ = [0.1,GaussNum(0.2,0.1),0.3,0.2]
u0 = [GaussNum(1.0,0.1),GaussNum(1.0,0.1)]
tspan = (0.,100.)
dt = 0.1
prob = ODEProblem(f,tspan,u0,θ)


isuncertain(x::GaussNum) = x.σ!=0
isuncertain(x::Number) = false
sig(x::GaussNum) = x.σ
mu(x::GaussNum) = x.μ

#struct UncertainODEProblem{OP<:ODEProblem,S0<:AbstractMatrix,S<:AbstractMatrix,X<:AbstractMatrix,I,J}
#    prob::OP
#    √Σ::S0
#    Xp::X
#    u_idx::I
#    θ_idx::J
#    function UncertainODEProblem(prob::OP) where OP<:ODEProblem
#        u_idx = findall(isuncertain, prob.u0)
#        θ_idx = findall(isuncertain, prob.θ)
#
#        uσ = [u.σ for u in prob.u0[u_idx]]
#        θσ = [θ.σ for θ in prob.θ[θ_idx]]
#        √Σ = Diagonal(vcat(uσ,θσ))
#
#        n = (length(uσ)+length(θσ))*2
#        Qp = sqrt(n) * [I(n) -I(n)]
#        Xp = vcat(prob.u0,prob.θ) .+ √Σ*Qp
#        Σ  = √Σ * √Σ'
#
#        new{OP,typeof(√Σ),typeof(u_idx),typeof(θ_idx)}(prob,√Σ,u_idx,θ_idx)
#    end
#end
struct UncertainODEProblem{OP,US,I,J} <: AbstractODEProblem
    prob::OP
    U0::US
    u_idx::I
    θ_idx::J
    function UncertainODEProblem(prob::ODEProblem)
        u_idx = findall(isuncertain, prob.u0)
        θ_idx = findall(isuncertain, prob.θ)
        idx = vcat(u_idx, length(prob.u0) .+ θ_idx)

        n = length(idx)
        Qp = hcat(map(idx) do i
            q = zeros(length(prob.u0)+length(prob.θ))
            q[i] = sqrt(n)
            q
        end, map(idx) do i
            q = zeros(length(prob.u0)+length(prob.θ))
            q[i] = -sqrt(n)
            q
        end)
        Qp = reduce(hcat,Qp)
        Σ_ = Diagonal(vcat(sig.(prob.u0), sig.(prob.θ)))
        μ0 = vcat(mu.(prob.u0),mu.(prob.θ)) .+ Σ_*Qp
        #U0 = μ0 .± diag(Σ_)

        prob = ODEProblem(prob.f, prob.tspan, mu.(prob.u0), mu.(prob.θ))

        new{typeof(prob),typeof(μ0),typeof(u_idx),typeof(θ_idx)}(prob,μ0,u_idx,θ_idx)
    end
end

nr_uncertainties(p::UncertainODEProblem) = length(p.u_idx)+length(p.θ_idx)

struct UncertainODESolver{S<:ODESolver} <: ODESolver
    solver::S
end

# function get_xp(p::UncertainODEProblem,i::Int)
#     xp = p.Xp[:,i]
#     u  = xp[p.u_idx]
#     θ  = p.prob.θ
#     θ[p.θ_idx] .= xp[length(p.u_idx)+1:end]
#     (u,θ)
# end


function setmean!(p::UncertainODEProblem, x)
    nu = length(p.prob.u0)
    p.prob.θ .= x[nu .+ (1:length(p.prob.θ))]
    u = x[1:nu]
end

function (s::UncertainODESolver)(p::UncertainODEProblem, μs, t)
    N = nr_uncertainties(p)
    μs = map(1:N) do i
        u = setmean!(p, μs[:,i])
        u = s.solver(p.prob, u, t)[1]
        vcat(u, p.prob.θ)
    end
    μs = reduce(hcat, μs) 
    μ = mean(μs,dims=2)
    Σ = Matrix((μs .- μ)*(μs .- μ)'/N)
    σ = sqrt.(diag(Σ))
    σ[p.u_idx] .= 0
    σ[p.θ_idx .+ length(p.prob.u0)] .= 0
    #μ .± σ, t+1
    μs, t+1
end

function solve(p::UncertainODEProblem, solver::UncertainODESolver)
    t = p.prob.tspan[1]; u = p.U0
    us = [u]; ts = [t]
    while t < prob.tspan[2]
        (u,t) = solver(p, u, t)
        push!(us,u)
        push!(ts,t)
    end
    ts, reduce(hcat,us)
end



uprob = UncertainODEProblem(prob)
solver = UncertainODESolver(RK2(0.2))
t, X = solve(uprob,solver)

# function solve(f,x0::AbstractVector,sqΣ0, θ,dt,N)
#     n = length(x0)
#     n2 = 2*length(x0)
#     Qp = sqrt(n)*[I(n) -I(n)]
# 
#     X = hcat([zero(x0) for i=1:N]...)
#     S = hcat([zero(x0) for i=1:N]...)
#     X[:,1]=x0
#     Xp = x0 .+ sqΣ0*Qp
#     sqΣ = sqΣ0
#     Σ = sqΣ* sqΣ'
#     S[:,1]= diag(Σ)
#     for t=1:N-1
#         for i=1:n2 # all quadrature points
#           Xp[:,i].=Xp[:,i] + dt*f(Xp[:,i],θ)
#         end
#         mXp=mean(Xp,dims=2)
#         X[:,t+1]=mXp
#         Σ=Matrix((Xp.-mXp)*(Xp.-mXp)'/n2)
#         S[:,t+1]=sqrt.(diag(Σ))
#         # @show Σ
# 
#     end
#     X,S
# end








