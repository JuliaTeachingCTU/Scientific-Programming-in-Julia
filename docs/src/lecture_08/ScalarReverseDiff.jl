mutable struct TrackedReal{T<:Real}
    data::T
    grad::Union{Nothing,T}
    children::Dict
    # this field is only need for printing the graph. you can safely remove it.
    name::String
end

track(x::Real,name="") = TrackedReal(x,nothing,Dict(),name)

function Base.show(io::IO, x::TrackedReal)
    t = isempty(x.name) ? "(tracked)" : "(tracked $(x.name))"
    print(io, "$(x.data) $t")
end

function accum!(x::TrackedReal)
    if isnothing(x.grad)
        x.grad = sum(accum!(v)*w for (v,w) in x.children)
    end
    x.grad
end

function gradient(f, args::Real...)
    ts = track.(args)
    y  = f(ts...)
    y.grad = 1.0
    accum!.(ts)
end


##########  RULES  #############################################################

function Base.:*(a::TrackedReal, b::TrackedReal)
    z = track(a.data * b.data, "*")
    a.children[z] = b.data  # dz/da=b
    b.children[z] = a.data  # dz/db=a
    z
end
function Base.:+(a::TrackedReal{T}, b::TrackedReal{T}) where T
    z = track(a.data + b.data, "+")
    a.children[z] = one(T)
    b.children[z] = one(T)
    z
end
function Base.sin(x::TrackedReal)
    z = track(sin(x.data), "sin")
    x.children[z] = cos(x.data)
    z
end


##########  Optimizion 2D function  ############################################

using Plots
g(x,y) = y*y + sin(x)
cscheme = cgrad(:RdYlBu_5, rev=true)
p1 = contour(-4:0.1:4, -2:0.1:2, g, fill=true, c=cscheme, xlabel="x", ylabel="y")
display(p1)

function descend(f::Function, λ::Real, args::Real...)
    Δargs = gradient(f, args...)
    args .- λ .* Δargs
end

function minimize(f::Function, args::T...; niters=20, λ=0.01) where T<:Real
    paths = ntuple(_->Vector{T}(undef,niters), length(args))
    for i in 1:niters
        args = descend(f, λ, args...)
        @info f(args...)
        for j in 1:length(args)
            paths[j][i] = args[j]
        end
    end
    paths
end

xs1, ys1 = minimize(g, 1.5, -2.4, λ=0.2, niters=34)
xs2, ys2 = minimize(g, 1.8, -2.4, λ=0.2, niters=16)

scatter!(p1, [xs1[1]], [ys1[1]], markercolor=:black, marker=:star, ms=7, label="Minimum")
scatter!(p1, [xs2[1]], [ys2[1]], markercolor=:black, marker=:star, ms=7, label=false)
scatter!(p1, [-π/2], [0], markercolor=:red, marker=:star, ms=7, label="Initial Point")
scatter!(p1, xs1[1:1], ys1[1:1], markercolor=:black, label="GD Path", xlims=(-4,4), ylims=(-2,2))

anim = @animate for i in 1:max(length(xs1), length(xs2))
    if i <= length(xs1)
        scatter!(p1, xs1[1:i], ys1[1:i], mc=:black, lw=3, xlims=(-4,4), ylims=(-2,2), label=false)
    end
    if i <= length(xs2)
        scatter!(p1, xs2[1:i], ys2[1:i], mc=:black, lw=3, label=false)
    end
    p1
end

gif(anim, "gd-path.gif", fps=15)
