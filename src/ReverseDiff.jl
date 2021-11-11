module ReverseDiff

using LinearAlgebra

export track, accum!, σ
export TrackedArray, TrackedMatrix, TrackedVector, TrackedReal

mutable struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    grad::Union{Nothing,A}
    children::Dict
end

const TrackedVector{T} = TrackedArray{T,1}
const TrackedMatrix{T} = TrackedArray{T,2}

mutable struct TrackedReal{T<:Real}
    data::T
    grad::Union{Nothing,T}
    children::Dict
end

track(x::Real) = TrackedReal(x, nothing, Dict())
data(x::TrackedReal) = x.data

track(x::Array) = TrackedArray(x, nothing, Dict())
data(x::TrackedArray) = x.data
for f in [:size, :length, :eltype]
	eval(:(Base.$(f)(x::TrackedArray, args...) = $(f)(x.data, args...)))
end
Base.getindex(x::TrackedArray, args...) = track(getindex(x.data,args...))
Base.show(io::IO, x::TrackedArray) = print(io, "Tracked $(x.data)")
Base.print_array(io::IO, x::TrackedArray) = Base.print_array(io, x.data)

track(x::Union{TrackedArray,TrackedReal}) = x
function reset!(x::Union{TrackedReal,TrackedArray})
    x.grad = nothing
    x.children = Dict()
    nothing
end

function accum!(x::Union{TrackedReal,TrackedArray})
    if isnothing(x.grad)
        x.grad = sum(λ(accum!(Δ)) for (Δ,λ) in x.children)
    end
    x.grad
end

function gradient(f, args::TrackedArray...)
    y = f(args...)
    (y isa TrackedReal) || error("Output of `f` must be a scalar.")
    y.grad = 1.0
    accum!.(args)
    Tuple(a.grad for a in args)
end


##########  RULES  #############################################################

function Base.sum(x::TrackedArray)
    z = track(sum(x.data))
    x.children[z] = Δ -> Δ*ones(eltype(x), size(x)...)
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

function Base.hcat(xs::TrackedArray...)
    y  = track(hcat(data.(xs)...))
    stops  = cumsum([size(x,2) for x in xs])
    starts = vcat([1], stops[1:end-1] .+ 1)
    for (start,stop,x) in zip(starts,stops,xs)
        x.children[y] = function (Δ)
            δ = if ndims(x) == 1
                Δ[:,start]
            else
                ds = map(_ -> :, size(x)) |> Base.tail |> Base.tail
                Δ[:, start:stop, ds...]
            end
            δ
        end
    end
    y
end

end
