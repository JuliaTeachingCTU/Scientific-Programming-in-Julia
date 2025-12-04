using OrdinaryDiffEq
using JLD2
#using Plots

function f(x,θ,t)
    α,β,γ,δ = θ
    x1,x2=x
    dx1 = α*x1 - β*x1*x2
    dx2 = δ*x1*x2 - γ*x2
    [dx1,dx2]
end

θ0 = [0.1,0.2,0.3,0.2]
x0 = [1.0,1.0]
tspan = (0., 100.)
prob = ODEProblem(f,x0,tspan,θ0)
sol = solve(prob,Tsit5())
#p = plot(sol)

dt = 0.2
ts = (tspan[1]):dt:(tspan[2])
us = reduce(hcat, sol(ts).u)

data = Dict(:u=>us, :t=>collect(ts))
jldsave("lotkadata.jld2", u=us, t=collect(ts))
