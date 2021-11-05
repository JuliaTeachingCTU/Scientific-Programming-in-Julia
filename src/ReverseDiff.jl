module ReverseDiff

using LinearAlgebra

export track, accum!, σ
export TrackedArray, TrackedMatrix, TrackedVector

mutable struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    grad::Union{Nothing,A}
    children::Dict
end

mutable struct Tracked{T<:Real}
    data::T
    grad::Union{Nothing,T}
    children::Dict
end

const TrackedVector{T} = TrackedArray{T,1}
const TrackedMatrix{T} = TrackedArray{T,2}

track(x::Real) = Tracked(x, nothing, Dict())
track(x::Array) = TrackedArray(x, nothing, Dict())

function reset!(x::Union{Tracked,TrackedArray})
    x.grad = nothing
    map(reset!, x.children |> keys |> collect)
    nothing
end

#Base.convert(::Type{Tracked}, x::Real) = track(x)
#Base.convert(::Type{TrackedArray}, x::Array) = track(x)
#Base.similar(x::TrackedArray, args...) = track(similar(x.data, args...))
#Base.BroadcastStyle(::Type{<:TrackedArray}) = Broadcast.ArrayStyle{TrackedArray}()
#function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TrackedArray}}, ::Type{T}) where T
#    track(similar(Array{T}, axes(bc)))
#end

for f in [:size, :length, :eltype]
	eval(:(Base.$(f)(x::TrackedArray, args...) = $(f)(x.data, args...)))
end
Base.getindex(x::TrackedArray, args...) = track(getindex(x.data,args...))
Base.show(io::IO, x::TrackedArray) = print(io, "Tracked $(x.data)")
Base.print_array(io::IO, x::TrackedArray) = Base.print_array(io, x.data)

function accum!(x::Union{Tracked,TrackedArray})
    if isnothing(x.grad)
        x.grad = sum(λ(accum!(Δ)) for (Δ,λ) in x.children)
    end
    x.grad
end

function gradient(f, args...)
    vs = track.(args)
    y  = f(vs...)
    (y.data isa Real) || error("Output of `f` must be a scalar.")
    y.grad = 1.0
    accum!.(vs)
end


##########  RULES  #############################################################

function Base.sum(x::TrackedArray)
    z = track(sum(x.data))
    x.children[z] = Δ -> ones(eltype(x), size(x)...)
    z
end

function Base.:*(X::TrackedArray, Y::TrackedArray)
    Z = track(X.data * Y.data)
    X.children[Z] = Δ -> Δ * Y.data'
    Y.children[Z] = Δ -> X.data' * Δ
    Z
end

function Base.:+(X::TrackedArray, Y::TrackedArray)
    Z = track(X.data + Y.data)
    X.children[Z] = Δ -> Δ
    Y.children[Z] = Δ -> Δ
    Z
end

function Base.:-(X::TrackedArray, Y::TrackedArray)
    Z = track(X.data - Y.data)
    X.children[Z] = Δ -> Δ
    Y.children[Z] = Δ -> -Δ
    Z
end

σ(x::Real) = 1/(1+exp(-x))
σ(x::AbstractArray) = σ.(x)
function σ(x::TrackedArray)
    z = track(σ(x.data))
    d = z.data
    x.children[z] = Δ -> Δ .* d .* (1 .- d)
    z
end

function Base.abs2(x::TrackedArray)
    y = track(abs2.(x.data))
    x.children[y] = Δ -> Δ .* 2x.data
    y
end

end
