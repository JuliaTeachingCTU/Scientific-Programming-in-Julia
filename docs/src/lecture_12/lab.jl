using Zygote

struct GaussNum{T<:Real} <: Real
    μ::T
    σ::T
end
Base.convert(::Type{T}, x::T) where T<:GaussNum = x
Base.convert(::Type{GaussNum{T}}, x::Number) where T = GaussNum(x,zero(T))
Base.promote_rule(::Type{GaussNum{T}}, ::Type{T}) where T = GaussNum{T}
Base.promote_rule(::Type{GaussNum{T}}, ::Type{GaussNum{T}}) where T = GaussNum{T}

# convert(GaussNum{Float64}, 1.0) |> display
# promote(GaussNum(1.0,1.0), 2.0) |> display
# error()

#+(x::GaussNum{T},a::T) where T =GaussNum(x.μ+a,x.σ)
#+(a::T,x::GaussNum{T}) where T =GaussNum(x.μ+a,x.σ)
#-(x::GaussNum{T},a::T) where T =GaussNum(x.μ-a,x.σ)
#-(a::T,x::GaussNum{T}) where T =GaussNum(x.μ-a,x.σ)
#*(x::GaussNum{T},a::T) where T =GaussNum(x.μ*a,a*x.σ)
#*(a::T,x::GaussNum{T}) where T =GaussNum(x.μ*a,a*x.σ)


# function Base.:*(x1::GaussNum, x2::GaussNum)
#     f(x1,x2) = x1 * x2
#     s1 = Zygote.gradient(μ -> f(μ,x2.μ), x1.μ)[1]^2 * x1.σ^2
#     s2 = Zygote.gradient(μ -> f(x1.μ,μ), x2.μ)[1]^2 * x2.σ^2
#     GaussNum(f(x1.μ,x2.μ), sqrt(s1+s2))
# end

function _uncertain(f, args::GaussNum...)
    μs  = [x.μ for x in args]
    dfs = Zygote.gradient(f,μs...)
    σ   = map(zip(dfs,args)) do (df,x)
        df^2 * x.σ^2
    end |> sum |> sqrt
    GaussNum(f(μs...), σ)
end

function _uncertain(expr::Expr)
    if expr.head == :call
        :(_uncertain($(expr.args[1]), $(expr.args[2:end]...)))
    else
        error("Expression has to be a :call")
    end
end

macro uncertain(expr)
    _uncertain(expr)
end

#register(mod, func) = :($mod.$func(args::GaussNum...) = _uncertain($func, args...))
function _register(func)
    mod = getmodule(eval(func))
    :($(mod).$(func)(args::GaussNum...) = _uncertain($func, args...)) |> eval
end
#register(mod::Nothing, func) = register(func)
macro register(func)
    _register(func) |> eval
end

f(x,y) = x+y*x

getmodule(f) = first(methods(f)).module

@register *
@register +
@register -
@register f

asdf(x1::GaussNum{T},x2::GaussNum{T}) where T =GaussNum(x1.μ*x2.μ, sqrt((x2.μ*x1.σ).^2 + (x1.μ * x2.σ).^2))
gggg(x1::GaussNum{T},x2::GaussNum{T}) where T =GaussNum(x1.μ+x2.μ, sqrt(x1.σ.^2 + x2.σ.^2))

x1 = GaussNum(rand(),rand())
x2 = GaussNum(rand(),rand())

display(x1*x2)
display(asdf(x1,x2))
display(_uncertain(*,x1,x2))
display(f(x1,x2))
display(@uncertain x1*x2)



using Plots
using JLD2

struct ODEProblem{F,T,U,P}
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

function (solver::Euler)(prob::ODEProblem, u, t)
    f, θ, dt  = prob.f, prob.θ, solver.dt
    (u + dt*f(u,θ), t+dt)
end

function (solver::RK2)(prob::ODEProblem, u, t)
    f, θ, dt  = prob.f, prob.θ, solver.dt
    uh = u + f(u,θ)*dt
    u + dt/2*(f(u,θ) + f(uh,θ)), t+dt
end


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

θ = [0.1,0.2,0.3,0.2]
u0 = [GaussNum(1.0,0.1),GaussNum(1.0,0.1)]
tspan = (0.,100.)
dt = 0.1
prob = ODEProblem(f,tspan,u0,θ)

t,X=solve(prob, RK2(0.2))
p1 = plot(t, X[1,:], label="x", lw=3)
plot!(p1, t, X[2,:], label="y", lw=3)

display(p1)
