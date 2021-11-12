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

function descend(f::Function, x0::Real, y0::Real; niters=20, λ=0.01)
    x, y = x0, y0
    ps = map(1:niters) do i
        (Δx, Δy) = gradient(f, x, y)
        x -= λ*Δx
        y -= λ*Δy
        [x,y]
    end |> ps -> reduce(hcat, ps)
    ps[1,:], ps[2,:]
end

xs1, ys1 = descend(g, 1.5, -2.4, λ=0.2, niters=34)
xs2, ys2 = descend(g, 1.8, -2.4, λ=0.2, niters=16)

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
