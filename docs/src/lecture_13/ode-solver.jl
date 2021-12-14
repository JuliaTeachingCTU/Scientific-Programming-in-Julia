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
    ts, us
end
