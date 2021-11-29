struct Dual{V<:Real} <: Real
    value::V
    partials::AbstractVector{V}
end
partials(d::Dual) = d.partials
value(d::Dual) = d.value

Dual(v::V,ps...) where V<:Real = Dual{V}(v,collect(ps))

function Base.show(io::IO, d::Dual{V}) where V
    print(io, "Dual(", d.value)
    for p in d.partials
        print(io,",",p)
    end
    print(io, ")")
end

function jacobian(f, x::AbstractVector{T}) where T
    duals = map(1:length(x)) do i
        ps = map(1:length(x)) do j
            i==j ? T(1) : T(0)
        end
        Dual(x[i], ps)
    end
    mapreduce(r->reshape(partials(r),1,:), vcat, f(duals))
end

##########  RULES  #############################################################

Base.:+(a::Dual, b::Dual) = Dual(a.value + b.value, a.partials + b.partials)
Base.:*(a::Dual, b::Dual) = Dual(a.value * b.value, b.value * a.partials + a.value * b.partials)

function cprod(x::AbstractVector)
    y = similar(x)
    y[1] = x[1]
    for i in 2:length(y)
        y[i] = y[i-1]*x[i]
    end
    y
end


x = collect(1:4)
#csum(x) |> display

dx = [Dual(x[1], 1, 0, 0),
      Dual(x[2], 0, 1, 0),
      Dual(x[3], 0, 0, 1)]
cprod(dx) |> display
csum(dx)

jacobian(cprod, x)
