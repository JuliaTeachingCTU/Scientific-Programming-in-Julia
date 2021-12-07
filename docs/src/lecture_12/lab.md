# Lab 12 - Differential Equations

In this lab you will implement a simple solver for *ordinary differential
equations* (ODE) as well as the two different methods for uncertainty
propagation that were introduced in the lecture.

## Euler ODE Solver

In this first part you will implement your own, simple, ODE framwork (feel free
to make it a package;) in which you can easily specify different ODE solvers.
The API is heavily inspired by [`DifferentialEquations.jl`](https://diffeq.sciml.ai/stable/),
so if you ever need to use it, you will already have a feeling for how it works.

Like in the lecture, we want to be able to specify an ODE like below.
```@example lab
function lotkavolterra(x,θ)
    α, β, γ, δ = θ
    x₁, x₂ = x

    dx₁ = α*x₁ - β*x₁*x₂
    dx₂ = δ*x₁*x₂ - γ*x₂

    [dx₁, dx₂]
end
nothing # hide
```
In the lecture we then solved it with a `solve` function that received all necessary
arguments to fully specify how the ODE should be solved. The number of parameters
there can quickly become very large, so we will introduce a new API for `solve`
which will always take only two arguments: `solve(::ODEProblem, ::ODESolver)`.
The `solve` function will only do some book-keeping and call the solver until
the ODE is solved for the full `tspan`.

The `ODEProblem` will contain all necessary parameters to fully specify the ODE
that should be solved. In our case that is the ODE `f` itself, initial
conditions `u0`, ODE parameters `θ`, and the time domain of the ODE `tspan`:
```@example lab
struct ODEProblem{F,T,U,P}
    f::F
    tspan::T
    u0::U
    θ::P
end
```

The solvers will all be subtyping the abstract type `ODESolver`. The `Euler` solver
from the lecture will need one field `dt` which specifies its time step:
```@example lab
abstract type ODESolver end

struct Euler{T} <: ODESolver
    dt::T
end
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Overload the call-method of `Euler` such that calling the solver with an `ODEProblem`
will perform one step of the Euler solver and returns updated ODE varialbes
`u1` and the corresponding timestep `t1`.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab
function (solver::Euler)(prob::ODEProblem, u, t)
    f, θ, dt  = prob.f, prob.θ, solver.dt
    (u + dt*f(u,θ), t+dt)
end
```
```@raw html
</p></details>
```
```@example lab
# define ODEProblem
θ = [0.1,0.2,0.3,0.2]
u0 = [1.0,1.0]
tspan = (0.,100.)
prob = ODEProblem(lotkavolterra,tspan,u0,θ)

# run one solver step
solver = Euler(0.2)
(u1,t1) = solver(prob,u0,0.)
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement the function `solve(::ODEProblem,::ODESolver)` which calls the solver
as many times as are necessary to solve the ODE for the full time domain.
`solve` should return a vector of timesteps and a corresponding matrix of
variables.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab
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
nothing # hide
```
```@raw html
</p></details>
```

You can load the true solution and compare it in a plot like below.  The file
that contains the correct solution is located here:
[`lotkadata.jld2`](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/master/docs/src/lecture_12/lotkadata.jld2).
```@example lab
using JLD2
using Plots

true_data = load("lotkadata.jld2")

p1 = plot(true_data["t"], true_data["u"][1,:], lw=4, ls=:dash, alpha=0.7, color=:gray, label="x Truth")
plot!(p1, true_data["t"], true_data["u"][2,:], lw=4, ls=:dash, alpha=0.7, color=:gray, label="y Truth")

(t,X) = solve(prob, Euler(0.2))

plot!(p1,t,X[1,:], color=1, lw=3, alpha=0.8, label="x Euler")
plot!(p1,t,X[2,:], color=2, lw=3, alpha=0.8, label="y Euler")
```

## Runge-Kutta ODE Solver

As you can see in the plot above, the Euler method quickly becomes quite
inaccurate. There exist many different ODE solvers. To demonstrate how we can
get significantly better results with a simple tweak, we will now implement the
second order Runge-Kutta method `RK2`:
```math
\begin{align*}
\tilde x_{n+1} &= x_n + hf(x_n, t_n)\\
       x_{n+1} &= x_n + \frac{h}{2}(f(x_n,t_n)+f(\tilde x_{n+1},t_{n+1}))
\end{align*}
```
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement the `RK2` solver according to the equations given above.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab
struct RK2{T} <: ODESolver
    dt::T
end
function (solver::RK2)(prob::ODEProblem, u, t)
    f, θ, dt  = prob.f, prob.θ, solver.dt
    uh = u + f(u,θ)*dt
    u + dt/2*(f(u,θ) + f(uh,θ)), t+dt
end
```
```@raw html
</p></details>
```

You should be able to use it exactly like our `Euler` solver before:
```@example lab
p1 = plot(true_data["t"], true_data["u"][1,:], lw=4, ls=:dash, alpha=0.7, color=:gray, label="x Truth")
plot!(p1, true_data["t"], true_data["u"][2,:], lw=4, ls=:dash, alpha=0.7, color=:gray, label="y Truth")

(t,X) = solve(prob, RK2(0.2))

plot!(p1,t,X[1,:], color=1, lw=3, alpha=0.8, label="x Euler")
plot!(p1,t,X[2,:], color=2, lw=3, alpha=0.8, label="y Euler")
```


# References

* [MIT18-330S12: Chapter 5](https://ocw.mit.edu/courses/mathematics/18-330-introduction-to-numerical-analysis-spring-2012/lecture-notes/MIT18_330S12_Chapter5.pdf)
