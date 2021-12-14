struct GaussNum{T<:Real} <: Real
    μ::T
    σ::T
end
StatsBase.mean(x::GaussNum) = x.μ
StatsBase.var(x::GaussNum) = x.σ^2
StatsBase.std(x::GaussNum) = x.σ
GaussNum(x,y) = GaussNum(promote(x,y)...)
±(x,y) = GaussNum(x,y)
Base.convert(::Type{T}, x::T) where T<:GaussNum = x
Base.convert(::Type{GaussNum{T}}, x::Number) where T = GaussNum(x,zero(T))
Base.promote_rule(::Type{GaussNum{T}}, ::Type{S}) where {T,S} = GaussNum{T}
Base.promote_rule(::Type{GaussNum{T}}, ::Type{GaussNum{T}}) where T = GaussNum{T}

gaussnums(x::MvNormal) = GaussNum.(mean(x), sqrt.(var(x)))
gaussnums(xs::Vector{<:MvNormal}) = reduce(hcat, gaussnums.(xs))
isuncertain(x::GaussNum) = x.σ!=0
isuncertain(x::Number) = false
