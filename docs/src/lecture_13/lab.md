# Lab 13 - Data-driven Differential Equations

In this lab you will implement your own _*Physics Informed Neural Network*_ (PINN)
(discussed in this weeks lecture) and an improved uncertainty propagation
based on the cubature rules from [lecture 12](@ref lec12).

_*Before your start*_: Please, install the necessary packages for this lab and
let them precompile while you familiarize yourself with the PINN implementation.
```julia
(@1.6) pkg> add Flux ForwardDiff Optim GalacticOptim Plots
```

## I. Physics Informed Neural Networks

You have already seen in the lecture that PINNs are implemented with a loss
function that contains two terms: the physics loss $\mathcal L_P$ and the data
loss $\mathcal L_D$ (which can also be used to define boundary conditions).
```math
\mathcal L = \mathcal L_P + \mathcal L_D
```
In this part of the lab we will implement both these loss terms step by step,
starting with the definition of the underlying neural network itself.

You have already seen the implementation of the so-called `FastLayer`s which
work with explicitly passed in parameters. The code below is adapted from the
package [`DiffEqFlux.jl`](https://github.com/SciML/DiffEqFlux.jl/blob/master/src/fast_layers.jl)
which you can use instead of precompiling `DiffEqFlux.jl` if you like.

```@example lab
using Flux

abstract type FastLayer <: Function end

struct FastDense{F,P} <: FastLayer
    in::Int
    out::Int
    σ::F
    initial_params::P
    function FastDense(in::Integer, out::Integer, σ=identity;
                       initW=Flux.glorot_uniform, initb=Flux.zeros32)
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

paramlength(c::FastChain) = sum(paramlength(x) for x in c.layers)
initial_params(c::FastChain) = vcat(initial_params.(c.layers)...)

(c::FastChain)(x,p) = applychain(c.layers, x, p)
applychain(::Tuple{}, x, p) = x
function applychain(fs::Tuple, x, p)
    x_ = first(fs)(x,p[1:paramlength(first(fs))])
    applychain(Base.tail(fs), x_, p[(paramlength(first(fs))+1):end])
end
nothing # hide
```
The `FastLayer`s can be used just like their implicitly parameterized relatives.
```@repl lab
hdim = 10;
u = FastChain(
    FastDense(1,hdim,tanh),
    FastDense(hdim,hdim,tanh),
    FastDense(hdim,2));
θ = initial_params(u);
u(rand(1,5),θ)
```

### The Physics Loss $\mathcal L_P$

PINNs can be used to solve differential equations (DE) by satisfying a
two-component loss function. The *physics loss* encourages the underlying
neural network to satisfy the given differential equation which is split into
a *left-hand side* $\text{LHS}$ and a *right-hand side* $\text{RHS}$:
```math
\mathcal L_P = \sum_{i=1}^N ||\text{LHS}(\hat x_i, \hat y_i, t_i) - \text{RHS}(\hat x_i, \hat y_i, t_i)||_2,
```
The symbols $\hat x$ and $\hat y$ denote the approximate solution of the DE
that comes from the neural network $u: \mathbb R \rightarrow \mathbb R^2$
which has one input (time $t$) and two outputs (DE variables $\hat x$ and $\hat
y$). In the Lotka-Volterra equations the LHS is just the first time derivative:
```math
\begin{align*}
  \dot x &= \alpha x - \beta xy,\\
  \dot y &= -\delta y + \gamma xy.
\end{align*}
```

Note that already inside $\mathcal L_P$ we have to compute a derivative of $u$
w.r.t to its input $t$. Hence, in order to perform gradient descent on the NN
parameters we will need higher-order derivatives. This is currently not possible
with `Zygote.jl` (but will be possible soon with `Diffractor.jl`), so we will
resort to `ForwardDiff.jl`. This will unfortunately cost us a little performance
because we have to iterate over batches of NN inputs, but for our small problem
this is ok.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Define a function to compute the derivative of a neural network `u` w.r.t to
one of its inputs, e.g. the second element of the input `x`
```julia
forward_derivative(u::FastLayer,x::AbstractArray,v::Int,θ)
```
You can compute derivatives of a many-to-many function by first computing
the Jacobian and then multiplying with a onehot vector that chooses the desired
row of the Jacobian. The function should accept both vector and batches of vectors
(i.e. matrices) as inputs `x`.

_*Hints*_:
- You can use `ForwardDiff.jacobian` to compute the Jacobian.
- You can use `Flux.onehot` for the onehot encoded vector.
- For derivatives of batches you can `map` over the columns of your input matrix.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab
using ForwardDiff

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
nothing # hide
```
```@raw html
</p></details>
```
```@repl lab
u = FastDense(2,3);
x = rand(2,5);
forward_derivative(u,x,2,initial_params(u))
```

Now we can define a new struct which will hold the network $u$, the LHS, and
the RHS of the PINN.
```@example lab
struct PINN{U<:FastLayer,L<:Function,R<:Function}
    u::U
    lhs::L
    rhs::R
end
```
The fields `lhs` and `rhs` are functions which are called like `lhs(u::FastLayer,x,θ)`.
The `lhs` function is simply the forward derivative of `u` w.r.t the first (and only)
input that represents the time:
```@example lab
lhs(u,x,θ) = forward_derivative(u,x,1,θ)
nothing # hide
```
The `rhs` calls the defined DE with the outputs of `u`. For simplicity we will
assume that the parameters of the Lotka-Volterra system are known, but they can
easily be incorporated in the vector of learned parameters `θ`.
Note that `lotkavolterra` is a vectorized version of the definition you know
from [lab 12](@ref lab12).
```@setup lab
function lotkavolterra(u::AbstractMatrix,θ)
    α, β, γ, δ = θ
    u₁, u₂ = u[1,:], u[2,:]

    du₁ = α.*u₁ - β.*u₁.*u₂
    du₂ = δ.*u₁.*u₂ - γ.*u₂

    [du₁ du₂]'
end
```
```@example lab
function lotkavolterra(u::AbstractMatrix,θ)
    # vectorized version of the lotkavolterra function from lab 12
end

lotka_params = [0.1f0, 0.2f0, 0.3f0, 0.2f0]
rhs(u,x,θ) = lotkavolterra(u(x,θ), lotka_params)
nothing # hide
```
With `lhs` and `rhs` in place we can construct the full PINN model:
```@example lab
hdim = 10
u = FastChain(FastDense(1,hdim,tanh), FastDense(hdim,hdim,tanh), FastDense(hdim,2))
θ = initial_params(u)
pinn = PINN(u,lhs,rhs)
nothing # hide
```
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
1. Define a function `physics_loss(m::PINN, x::AbstractArray, θ::AbstractVector)`
   which implements $\mathcal L_P$. The input `x` can either be a vector of input
   coordinates (in our case there is only time, so the input would be e.g. `[1.0]` for
   a single example), or the input can be a batch of coordinates (i.e. a matrix).
2. Implement the vectorized version of the `lotkavolterra` function.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab
using StatsBase

function physics_loss(m::PINN, x::AbstractArray, θ::AbstractVector)
    u = m.u
    mean(abs2, m.lhs(u,x,θ) - m.rhs(u,x,θ))
end

function lotkavolterra(u::AbstractMatrix,θ)
    α, β, γ, δ = θ
    u₁, u₂ = u[1,:], u[2,:]

    du₁ = α.*u₁ - β.*u₁.*u₂
    du₂ = δ.*u₁.*u₂ - γ.*u₂

    [du₁ du₂]'
end
nothing # hide
```
```@raw html
</p></details>
```
You test that your physics loss works by computing a gradient for some random
inputs
```@repl lab
testloss(θ) = physics_loss(pinn, rand(1,10), θ)
testloss(θ)
ForwardDiff.gradient(testloss,θ)
```


### The Data Loss $\mathcal L_D$

To fully specify a DE we have to specify its boundary conditions.  In PINNs, a
boundary condition is conceptually the same as incorporating data, so we will
implement any potential boundary condition by defining a data loss $\mathcal L_D$
```math
\mathcal L_D = \sum_{i=1}^M ||\hat u(x_i) - y_i||_2,
```
where $x_i$ are input coordinates and $y_i$ corresponding labels.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Define a function which implements the data loss $\mathcal L_D$ from the equation
above.
```julia
data_loss(m::PINN, x::AbstractArray, y::AbstractArray, θ::AbstractVector)
```
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab
function data_loss(m::PINN, x::AbstractArray, y::AbstractArray, θ::AbstractVector)
    mean(abs2, m.u(x,θ) - y)
end
nothing # hide
```
```@raw html
</p></details>
```


### PINN Training

To use our PINN we now just need to load some data (or define boundary
conditions) and define a training grid for the physics loss.
To make the training reasonably short we will fit one bump of the Lotka-Volterra
system which corresponds to a `tspan = (0,50)`, so we can define a time grid
with
```@repl lab
xs_grid = reshape(collect(0f0:2f0:50f0), 1, :);
physics_loss(pinn,xs_grid,θ)
```
Note that we are using `Float32` to take advantage of faster training of our
`Float32` model.

To show how little data a PINN needs we can pick only three data points from
our `loktadata.jld2`
```@repl lab
using JLD2
data = load("../lecture_12/lotkadata.jld2");
xs = Float32.(data["t"][1:100:201]);
xs = reshape(xs, 1, :)
ys = Float32.(data["u"][:,1:100:201])
data_loss(pinn,xs,ys,θ)
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement a closure `plotprogress(θ)` which closes over the global variables
`pinn`, `xs`, `ys`, and `data`, and plots the correct solution
(obtained from `data`), the data that is used for training (`xs` and `ys`), and
the current solution of the PINN (based on `θ`).
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab
using Plots

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
    plot!(p1, xlim=(xs_grid[1],xs_grid[end]))
end
nothing # hide
```
```@raw html
</p></details>
```
Your plot can (but does not have to) look like this
```@example lab
plotprogress(θ)
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
To optimize our PINN we will use `GalacticOptim.solve`, which expects a loss
function with the API `loss(θ,p=nothing)` and returns a tuple `(l, ls)` with
the final loss value `l`, and another tuple `ls` that contains optional diagnostic
values. We will stick our individual data/physics loss values in there.

Additionally, we can define a `callback(l,ls,θ)` which prints/plots our
training progress. `l` is the final loss, `ls` the diagnostics, and `θ` our current
optimization parameters. The callback should always return `false` (this can
be used for early stopping).

1. Define the final PINN `loss(θ,p=nothing)` function.
2. Define a `callback(l,ls,θ)` function.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab
function loss(θ,p=nothing)
    l1 = internal_loss(pinn, xs_grid, θ)
    l2 = data_loss(pinn, xs, ys, θ)
    l1+l2, (l1,l2)
end

function callback(θ,l,ls)
    (internal, dataloss) = ls
    plotprogress(pinn,θ,xs,ys) |> display
    @info l internal dataloss
    false
end
nothing # hide
```
```@raw html
</p></details>
```

To start training with `GalacticOptim.jl` we just need to define which AD backend
to use and pass in our constructed loss
```julia
func = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(func, θ)
opt  = LBFGS()
solve(prob, opt, maxiters=10_000, cb=Flux.throttle(callback,1))
```
![pinn](pinn.png)


## II. Uncertainy Prop. With Cubature Rules

In [lecture 12](@ref lec12) we have introduced an advanced uncertainty propagation
based on cubature rules. In order to implement the cubature rules we only need
to solve our ODE for the set of $\sigma$-points `Xp`. If we construct a new
`GaussODEProblem` we can make use of our already implemented ODE solvers
(which you can find
[here](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/master/docs/src/lecture_13/lab/ode-solver.jl)),
and for plotting our `GaussNum`s
([here](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/master/docs/src/lecture_13/lab/gaussnum.jl)).

Each $\sigma$-point will contain uncertain initial conditions, and uncertain
parameters. So if we define the inputs to a `GaussODEProblem` as below
```@example cub
include("lab/gaussnum.jl")
# there are two tiny changes since the last lab in here. take a look a the
# "NOTE"s in the file below if you like.
include("lab/ode-solver.jl")

function lotkavolterra(x,θ)
    α, β, γ, δ = θ
    x₁, x₂ = x

    dx₁ = α*x₁ - β*x₁*x₂
    dx₂ = δ*x₁*x₂ - γ*x₂

    [dx₁, dx₂]
end

θ = [0.1±0.01, 0.2, 0.3, 0.2]
u0 = [1.0±0.2, 1.0±0.2]
tspan = (0., 100.)
nothing # hide
```
we want a set of 6 $\sigma$-points (2 for each of the 3 uncertain parameters).
The initial $\sigma$-points will be stored (as a matrix) in the `u0` field of
our new `GaussODEProblem`:

```@example cub
struct GaussODEProblem{F,T<:Tuple{Number,Number},U<:AbstractMatrix,P<:AbstractVector,I} <: AbstractODEProblem
    f::F      # RHS of ODE
    tspan::T
    u0::U     # initial σ-points
    θ::P      # ODE parameters (will be mutated)
    θ_idx::I  # indices of uncertain parameters
    size::Int # state size of the ODE
end
Base.size(prob::GaussODEProblem) = size(prob.u0,1)
statesize(prob::GaussODEProblem) = prob.size
nothing # hide
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement a constructor for `GaussODEProblem` with the following type signature
```julia
GaussODEProblem(f,tspan,u0::V,θ::V) where V<:AbstractVector{<:GaussNum}
```
which computes the $\sigma$-points, saves the indices of the uncertain ODE parameters
as well as the state size of the ODE.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example cub
using LinearAlgebra

function GaussODEProblem(f,tspan,u0::V,θ::V) where V<:AbstractVector{<:GaussNum}
    θ_idx = findall(isuncertain, θ)

    μ = vcat(mean.(u0), mean.(θ[θ_idx]))
    d = length(μ)
    Σ = vcat(var.(u0), var.(θ[θ_idx])) |> diagm
    Σ½ = cholesky(Σ).L
    Qp = sqrt(d)*[I(d) -I(d)]
    Xp = μ .+ Σ½*Qp

    GaussODEProblem(f,tspan,Xp,mean.(θ),θ_idx,length(u0))
end
nothing # hide
```
```@raw html
</p></details>
```
You should be able to construct the `GaussODEProblem` as follows
```@repl cub
prob = GaussODEProblem(lotkavolterra,tspan,u0,θ)
prob.u0
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement the call method of a new `GaussODESolver`
```@example cub
struct GaussODESolver{S<:ODESolver} <: ODESolver
    solver::S
end
nothing # hide
```
which, in every call, iterates over the $\sigma$-points and returns updated
$\sigma$-points and an updated time step.  While you compute new
$\sigma$-points you can make use of the classical solver inside `GaussODESolver`.
The only thing you need to do is set the uncertain ODE parameters to the correct
values from your current $\sigma$-point.

_*Hint*_:
You can implement a function `setmean!(prob::GaussODEProblem,xi)` which takes
a $\sigma$-point `xi`, mutates `prob.θ`, and returns only the part of `xi` that
describes the ODE state.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example cub
function setmean!(prob::GaussODEProblem,xi)
    prob.θ[prob.θ_idx] .= xi[(statesize(prob)+1):end]
    xi[1:statesize(prob)]
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
```
```@raw html
</p></details>
```
Your solver should now work like this:
```@repl cub
solver = GaussODESolver(RK2(0.2))
solver(prob, prob.u0, 0.)[1]
```
To compare with the MC result we can now solve our `GaussODEProblem` and
a few samples of a normal `ODEProblem` and compare them.
```@example cub
using JLD2
raw_data = load("../lecture_12/lotkadata.jld2")

t, us = solve(prob,solver)

# compute GaussNums and plot them
gus = map(us) do u
    u = u[1:statesize(prob),:]
    μ = mean(u,dims=2) |> vec
    σ = std(u,dims=2) |> vec
    GaussNum.(μ,σ)
end
gus = reduce(hcat,gus)
p1 = plot(raw_data["t"], raw_data["u"][1,:], c=3, lw=3, alpha=0.7, label="correct")
plot!(p1, raw_data["t"], raw_data["u"][2,:], c=3, lw=3, alpha=0.7, label=false)
plot!(p1, t, gus[1,:], c=1, lw=3, label="x")
plot!(p1, t, gus[2,:], c=2, lw=3, label="y")


# define ODEProblem sampling
function Base.rand(x::AbstractVector{<:GaussNum{T}}) where T
    mean.(x) .+ std.(x) .* randn(T,length(x))
end
Base.rand(prob::ODEProblem) = ODEProblem(prob.f, prob.tspan, rand(prob.u0), rand(prob.θ))

# compute & plot MC samples
prob = ODEProblem(lotkavolterra,tspan,u0,θ)
p2 = plot(raw_data["t"], raw_data["u"][1,:], c=3, lw=3, alpha=0.7, label="correct")
plot!(p2, raw_data["t"], raw_data["u"][2,:], c=3, lw=3, alpha=0.7, label=false)
Us = map(1:200) do i
    t, us = solve(rand(prob),solver.solver)
    us = reduce(hcat,us)
    plot!(p2, t, us[1,:], c=1, alpha=0.1, label=false)
    plot!(p2, t, us[2,:], c=2, alpha=0.1, label=false)
    us
end
plot!(p2, t, GaussNum.(mean(Us)[1,:],std(Us)[1,:]), c=1, lw=3, label="x")
plot!(p2, t, GaussNum.(mean(Us)[2,:],std(Us)[2,:]), c=2, lw=3, label="y")
plot(p1,p2,size=(800,300))
```
You can see that the cubature rules come quite close to the "correct" MC
solution.  However, in both cases the mean starts to deviate quite significatly
from the expected behaviour (and will approach an average for $x$ and $y$ for
longer times).


## Data Assimilation in Uncertain ODEs

If we have a few measurements available (say every 100 steps) we can make use
of them by reprojecting our current $\sigma$-points with the help of a
measurement. This can be implemented via *Bayesian Filtering* as discussed in the
lecture.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement a function `filter(prob::GaussODEProblem, solver::GaussODESolver,
data, σy::Real)` which performs Baysian filtering.  The `data` are assumed to
be tuples of timestamps and measurements `(ty::Real,y::Real)`.  Iterate over
the data, perform `solver` steps until you reach the timestamp `ty` and then
apply the filtering equations from the lecture.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example cub
function filter(prob::GaussODEProblem, solver::GaussODESolver, data, σy)
    t = prob.tspan[1]; u = prob.u0
    us = [u]; ts = [t]

    for (ty, y) in data
        while t <= ty
            (u,t) = solver(prob, u, t)
            push!(us,u)
            push!(ts,t)
        end

        yₚ = u[1,:] # prediction. we measure only the first variable
        μᵧ = mean(yₚ)
        μᵤ = mean(u, dims=2)

        Σᵧᵧ = cov(yₚ, corrected=false) + σy
        Σᵤᵤ = cov(u, dims=2, corrected=false)

        Σᵤᵧ = cov(u, yₚ, dims=2, corrected=false)
        K   = Σᵤᵧ*inv(Σᵧᵧ)

        μ = vec(μᵤ + K*(y .- μᵧ))
        Σ = Hermitian(Σᵤᵤ - K*Σᵤᵧ')

        d = size(prob)
        Qp = sqrt(d)*[I(d) -I(d)]
        u = μ .+ cholesky(Σ).L * Qp
    end
    ts, us
end
nothing # hide
```
```@raw html
</p></details>
```
Now you can use `filter` instead of `solve` to incoorporate some data points
every 100 steps.
```@example cub
# define ODE problem
θ = [0.1±0.01, 0.2, 0.3, 0.2]
u0 = [1.0±0.2, 1.0±0.2]
tspan = (0., 100.)
prob = GaussODEProblem(lotkavolterra,tspan,u0,θ)
solver = GaussODESolver(RK2(0.1))

# collect data
using JLD2
raw_data = load("../lecture_12/lotkadata.jld2")
ts = Float32.(raw_data["t"][1:100:end])
ys = Float32.(raw_data["u"][:,1:100:end])
data = [(t,y[1]) for (t,y) in zip(ts,eachcol(ys))]
t, us = filter(prob, solver, data, 0.01)

# plot results
gus = map(us) do u
    u = u[1:statesize(prob),:]
    μ = mean(u,dims=2) |> vec
    σ = std(u,dims=2) |> vec
    GaussNum.(μ,σ)
end
gus = reduce(hcat,gus)
p1 = plot(t, gus[1,:], lw=3, label="x")
plot!(p1, t, gus[2,:], lw=3, label="y")
plot!(p1, raw_data["t"], raw_data["u"][1,:], c=3, lw=3, alpha=0.6, label="correct")
plot!(p1, raw_data["t"], raw_data["u"][2,:], c=3, lw=3, alpha=0.6, label=false)
```
You can see that the mean and the variance is reprojected every 100 steps and
we stay much closer to our perfect solution.
