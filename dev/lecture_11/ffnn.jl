using GLMakie
function flower(n; npetals = 8)
    n = div(n, npetals)
    x = mapreduce(hcat, (1:npetals) .* (2π/npetals)) do θ
        ct = cos(θ)
        st = sin(θ)

        x0 = tanh.(randn(1, n) .- 1) .+ 4.0 .+ 0.05.* randn(1, n)
        y0 = randn(1, n) .* 0.3

        x₁ = x0 * cos(θ) .- y0 * sin(θ)
        x₂ = x0 * sin(θ) .+ y0 * cos(θ)
        vcat(x₁, x₂)
    end
    _y = mapreduce(i -> fill(i, n), vcat, 1:npetals)
    y = zeros(npetals, length(_y))
    foreach(i -> y[_y[i], i] = 1, 1:length(_y))
    Float32.(x), Float32.(y)
end

x, y = flower(900)
scatter(x[1,:], x[2,:], color = mapslices(argmax, y, dims = 1)[:])

#######
#   Define a Tracked Array for operator overloading AD
#######
struct TrackedArray{T,N,V<:AbstractArray{T,N}} <: AbstractArray{T,N}
    value::V
    deriv::Union{Nothing,V}
    tape::Vector{Any}
end

TrackedArray(a::AbstractArray) = TrackedArray(a, similar(a) .= 0, [])
TrackedMatrix{T,V} = TrackedArray{T,2,V} where {T,V<:AbstractMatrix{T}}
TrackedVector{T,V} = TrackedArray{T,1,V} where {T,V<:AbstractVector{T}}
Base.size(a::TrackedArray) = size(a.value)
Base.show(io::IO, ::MIME"text/plain", a::TrackedArray) = show(io, a)
Base.show(io::IO, a::TrackedArray) = print(io, "TrackedArray($(size(a.value)))")
value(A::TrackedArray) = A.value
resetgrad!(A::TrackedArray) = (A.deriv .= 0; empty!(A.tape))
value(A) = A
track(A) = TrackedArray(A)
track(a::Number) = TrackedArray(reshape([a], 1, 1))

function accum!(A::TrackedArray)
    isempty(A.tape) && return(A.deriv)
    A.deriv .= sum(g(accum!(r)) for (r, g) in A.tape)
    empty!(A.tape)
    A.deriv
end

#######
#   Define AD rules for few operations appearing in FFNN
#######
import Base: +, *
import Base.Broadcast: broadcasted
function *(A::TrackedMatrix, B::TrackedMatrix)
    a, b = value.((A, B))
    C = track(a * b)
    push!(A.tape, (C, Δ -> Δ * b'))
    push!(B.tape, (C, Δ -> a' * Δ))
    C
end

function *(A::TrackedMatrix, B::AbstractMatrix)
    a, b = value.((A, B))
    C = track(a * b)
    push!(A.tape, (C, Δ -> Δ * b'))
    C
end

function broadcasted(::typeof(+), A::TrackedMatrix, B::TrackedVector)
    C = track(value(A) .+ value(B))
    push!(A.tape, (C, Δ -> Δ))
    push!(B.tape, (C, Δ -> sum(Δ, dims = 2)[:]))
    C
end

function σ(x::Real) 
    t = @fastmath exp(-abs(x))
    y = ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
    ifelse(x > 40, one(y), ifelse(x < -80, zero(y), y))
end

broadcasted(::typeof(identity), A::TrackedArray) = A

function broadcasted(::typeof(σ), A::TrackedArray)
    Ω = σ.(value(A))
    C = track(Ω)
    push!(A.tape, (C, Δ -> Δ .* Ω .* (1 .- Ω)))
    C
end

function mse(A::TrackedMatrix, B::AbstractMatrix)
    n = size(A, 2)
    a = value(A)
    c = similar(a, 1, 1)
    c .= sum((a .- B).^2)/2n
    C = track(c)
    push!(A.tape, (C, Δ -> Δ .* (a .- B) ./ n))
    C
end

mse(x::AbstractMatrix, y::AbstractMatrix) = sum((x - y).^2) / (2*size(x,2))

#######
#   Define a Dense layer
#######
struct Dense{F,W,B}
    σ::F 
    w::W 
    b::B
end

Base.show(io::IO, m::Dense) = print(io, "Dense($(size(m.w,2)) → $(size(m.w,1)))")
Dense(i::Int, o::Int, σ = identity) = Dense(σ, randn(Float32, o, i), randn(Float32, o))
track(m::Dense) = Dense(m.σ, track(m.w), track(m.b))
track(m::ComposedFunction) = track(m.outer) ∘ track(m.inner)
(m::Dense)(x) = m.σ.(m.w * x .+ m.b)
params(m::ComposedFunction) = vcat(params(m.outer), params(m.inner))
params(m::Dense) = [m.w, m.b]

#######
#   Let's try to actually train a model
#######
x, y = flower(900)
function initmodel()
    m₁ = track(Dense(2, 20, σ))
    m₂ = track(Dense(20, 20, σ))
    m₃ = track(Dense(20, size(y,1)))
    m = m₃ ∘ m₂ ∘ m₁
end
m = initmodel()
m(x) |> value 

######
#   Let's try to learn the parameters
######
α = 0.01
ps = params(m)
@elapsed for i in 1:10000
    foreach(resetgrad!, ps)
    loss = mse(m(x), y)
    fill!(loss.deriv, 1)
    foreach(accum!, ps)
    foreach(x -> x.value .-= α .* x.deriv, ps)
    mod(i,250) == 0 && println("loss after $(i) iterations = ", sum(value(loss)))
end

all(mapslices(argmax, value(m(x)), dims = 1)[:] .== mapslices(argmax, y, dims = 1)[:])
scatter(x[1,:], x[2,:], color = mapslices(argmax, value(m(x)), dims = 1)[:])

######
#   Let's try to move the computation to GPU
######
using CUDA
gpu(x::AbstractArray) = CuArray(x)
gpu(x::TrackedArray) = TrackedArray(CuArray(value(x)))
gpu(m::Dense) = Dense(m.σ, gpu(m.w), gpu(m.b))
gpu(m::ComposedFunction) = gpu(m.outer) ∘ gpu(m.inner)

gx, gy = gpu(x), gpu(y)
m = gpu(m)
ps = params(m)
@elapsed for i in 1:10000
    foreach(resetgrad!, ps)
    loss = mse(m(gx), gy)
    fill!(loss.deriv, 1)
    foreach(accum!, ps)
    foreach(x -> x.value .-= α .* x.deriv, ps)
    mod(i,250) == 0 && println("loss after $(i) iterations = ", sum(value(loss)))
end

#######
#   Why we see a small speed-up? The problem is small
#######
using BenchmarkTools
p = randn(Float32, 20, 2)
@benchmark $(p) * $(x)
gp = gpu(p)
@benchmark $(gp) * $(gx)


######
#   Let's verify the gradients
######
using FiniteDifferences
ps = [m₃.w, m₃.b, m₂.w, m₂.b, m₁.w, m₁.b]
map(ps) do p 
    foreach(resetgrad!, ps)
    loss = mse(m(x), y)
    fill!(loss.deriv, 1)
    foreach(accum!, ps)
    accum!(p)
    θ = deepcopy(value(p))
    Δθ = deepcopy(p.deriv)
    f = θ -> begin
        p.value .= θ
        value(mse(m(x), y))
    end
    sum(abs2.(grad(central_fdm(5, 1), f, θ)[1] - Δθ))
end


