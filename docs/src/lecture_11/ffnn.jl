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
    foreach(i -> y[_y[i], i] = 1, _y)
    Float32.(x), Float32.(y)
end

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

relu(x::Real) = max(0,x)

function broadcasted(::typeof(identity), A::TrackedArray)
    C = track(value(A))
    push!(A.tape, (C, Δ -> Δ))
    C
end

function broadcasted(::typeof(relu), A::TrackedArray)
    C = track(relu.(value(A)))
    push!(A.tape, (C, Δ -> Δ .* value(A) .> 0))
    C
end

function mse(A::TrackedMatrix, B::AbstractMatrix)
    n = size(A, 1)
    a = value(A)
    C = track(sum((a .- B).^2)/2)
    push!(A.tape, (C, Δ -> (a .- B)))
    C
end

mse(x::AbstractMatrix, y::AbstractMatrix) = sum((x - y).^2) / (2n)

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

#######
#   Let's try to actually train a model
#######
x, y = flower(900)
m₁ = track(Dense(2, 20, relu))
m₂ = track(Dense(20, 20, relu))
m₃ = track(Dense(20, size(y,1)))
m = m₃ ∘ m₂ ∘ m₁
m(x) |> value 

######
#   Let's try to learn the parameters
######
α = 0.001
ps = [m₃.w, m₃.b, m₂.w, m₂.b, m₁.w, m₁.b]
for i in 1:1000
    loss = mse(m(x), y)
    fill!(loss.deriv, 1)
    foreach(accum!, ps)
    foreach(x -> x.value .-= α .* x.deriv, ps)
    foreach(x -> x.deriv .= 0, ps)
    mod(i,100) == 0 && println("loss after $(i) iterations = ", value(loss)[])
end

using CUDA
gpu(x::AbstractArray) = CuArray(x)
gpu(x::TrackedArray) = TrackedArray(CuArray(value(x)))
gpu(m::Dense) = Dense(m.σ, gpu(m.w), gpu(m.b))

gpu(m)(gpu(x))


