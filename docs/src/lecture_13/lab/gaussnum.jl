using Statistics

struct GaussNum{T<:Real} <: Real
    μ::T
    σ::T
end
Statistics.mean(x::GaussNum) = x.μ
Statistics.var(x::GaussNum) = x.σ^2
Statistics.std(x::GaussNum) = x.σ
GaussNum(x,y) = GaussNum(promote(x,y)...)
±(x,y) = GaussNum(x,y)
Base.convert(::Type{T}, x::T) where T<:GaussNum = x
Base.convert(::Type{GaussNum{T}}, x::Number) where T = GaussNum(x,zero(T))
Base.promote_rule(::Type{GaussNum{T}}, ::Type{S}) where {T,S} = GaussNum{T}
Base.promote_rule(::Type{GaussNum{T}}, ::Type{GaussNum{T}}) where T = GaussNum{T}

isuncertain(x::GaussNum) = x.σ!=0
isuncertain(x::Number) = false

# TODO: add AD/arithmetic funcs

using Plots
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
        fillcolor := :gray
        fillalpha := 0.5
        fillrange := μs .- σs
        # ensure no markers are shown for the error band
        markershape := :none
        # return series data
        ts, μs .+ σs
    end
    ts, μs
end
