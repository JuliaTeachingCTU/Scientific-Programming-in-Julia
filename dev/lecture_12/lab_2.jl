abstract type AbstractODEProblem end

struct ODEProblem{F,T,U,P} <: AbstractODEProblem
    f::F
    tspan::T
    u0::U
    θ::P
end

abstract type ODESolver end
struct Euler{T} <: ODESolver
    dt::T
end
struct RK2{T} <: ODESolver
    dt::T
end

function f(x,θ)
    α, β, γ, δ = θ
    x₁, x₂ = x

    dx₁ = α*x₁ - β*x₁*x₂
    dx₂ = δ*x₁*x₂ - γ*x₂

    [dx₁, dx₂]
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

function (solver::Euler)(prob::ODEProblem, u, t)
    f, θ, dt  = prob.f, prob.θ, solver.dt
    (u + dt*f(u,θ), t+dt)
end

function (solver::RK2)(prob::ODEProblem, u, t)
    f, θ, dt  = prob.f, prob.θ, solver.dt
    uh = u + f(u,θ)*dt
    u + dt/2*(f(u,θ) + f(uh,θ)), t+dt
end


θ = [0.1,0.2,0.3,0.2]
u0 = [1.0,1.0]
tspan = (0.,100.)
dt = 0.1
prob = ODEProblem(f,tspan,u0,θ)

t,X=solve(prob, RK2(0.2))

using Plots
p1 = plot(t, X[1,:], label="x", lw=3)
plot!(p1, t, X[2,:], label="y", lw=3)

display(p1)

#
θ = [0.2,0.2,0.3,0.2]
u0 = [1.0,1.0]
tspan = (0.,100.)
dt = 0.1
prob2 = ODEProblem(f,tspan,u0,θ)

t,X2=solve(prob2, RK2(0.2))

using Optim

function loss(θin,prob::ODEProblem,Y)
    prob.θ.=θin
    t,Xn=solve(prob,RK2(0.2))
    sum((Y.-Xn).^2)
end
θopt = copy(θ)
O=Optim.optimize(θ->loss(θ,prob,X),θopt)
O=Optim.optimize(θ->loss(θ,prob,X),θopt,LBFGS())

using DiffEqFlux
nn=FastDense(2,2)
p = initial_params(nn)
nn([1,2],p)


function fy(x,θ)
    α, β, γ, δ, ω = θ
    x₁, x₂ = x

    dx₁ = α*x₁ - β*x₁*x₂ + ω*x₂
    dx₂ = δ*x₁*x₂ - γ*x₂

    [dx₁, dx₂]
end

#
θy = [0.2,0.2,0.3,0.2,0.1]
u0 = [1.0,1.0]
tspan = (0.,100.)
dt = 0.1
proby = ODEProblem(fy,tspan,u0,θy)

t,Xy=solve(proby, RK2(0.2))

py = plot(t, Xy[1,:], label="x", lw=3)
plot!(py, t, Xy[2,:], label="y", lw=3)
savefig("LV_omega.svg")

function fnn(x,θ)
    α, β, γ, δ = θ[1:4]
    x₁, x₂ = x

    dx₁ = α*x₁ - β*x₁*x₂ 
    dx₂ = δ*x₁*x₂ - γ*x₂

    [dx₁, dx₂]+nn(x,@view θ[5:end])
end

θnn = [0.2,0.2,0.3,0.2,0.01*initial_params(nn)...]
probnn = ODEProblem(fnn,tspan,u0,θnn)

θopt = copy(θnn)
O=Optim.optimize(θ->loss(θ,probnn,Xy),θopt)
