```
    using ChainRulesCore, ChainRules, ChainRulesTestUtils

    Write custom `rrule` for following function $f(x,y) = x^2 + 3y$.


    f1(x::T, y::T) where T<: Real = x^2 + 3*y

    You can test your solution using

    test_rrule(f1, randn(), randn())
```
using ChainRulesCore, ChainRules, ChainRulesTestUtils
using Zygote


# Ecersion 1: Implementing rrule for a simple function
f1(x::T, y::T) where T<: Real = x^2 + 3*y

function ChainRulesCore.rrule(::typeof(f1), x::T, y::T) where T<: Real
    y = f1(x, y)
    function pullback(ȳ)
        # return (tangent_for_function, tangent_for_x, tangent_for_y)
        return NoTangent(), 2x * ȳ, 3 * ȳ
    end
    return y, pullback
end

gradient((a,b)->a^2 + 3*b, 2f0, 4f0)
gradient((a,b)->f1(a,b), 2f0, 4f0)


# Exercise
mymaximum(x) = maximum(x)

function ChainRulesCore.rrule(::typeof(mymaximum), x::AbstractArray{T}) where T
    y, idx = findmax(x)

    function pullback(ȳ)
        x̄ = zero(x)        # same shape as x, typed zeros
        x̄[idx] += ȳ        # only the max index gets the incoming gradient
        return NoTangent(), x̄
    end

    return y, pullback
end

test_rrule(mymaximum, randn(10));
test_rrule(mymaximum, randn(10,10));



# Exercise 3: Implementing rrule for 2D pooling operation
create_range(len::Int, step::Int) = [i:min(i + step - 1, len) for i in 1:step:len]

AUR = Vector{UnitRange{Int64}}

x = randn(10,10);
s1 = create_range(10, 3);
s2 = create_range(10, 2);

pool_native(x::AbstractArray, seg₁::AUR, seg₂::AUR) = [sum(x[sᵢ, sⱼ]) for sᵢ in seg₁, sⱼ in seg₂]
gradient(a->sum(pool_native(a, s1, s2)), x)[1]

function pool_sum(x::AbstractArray, seg₁::AUR, seg₂::AUR)
    y = similar(x, length(seg₁), length(seg₂))
    for (i, sᵢ) in enumerate(seg₁)
        for (j, sⱼ) in enumerate(seg₂)
            y[i,j] = sum(x[sᵢ, sⱼ]) 
        end
    end
    return y
end


function ChainRulesCore.rrule(::typeof(pool_sum), x::AbstractArray, seg₁::AUR, seg₂::AUR)
    y = pool_sum(x, seg₁, seg₂)

    function pool_sum_pullback(ȳ)
        x̄ = zero(x)
        for (i, sᵢ) in enumerate(seg₁)
            for (j, sⱼ) in enumerate(seg₂)
                x̄[sᵢ, sⱼ] .+= ȳ[i,j]
            end
        end
        return NoTangent(), x̄, NoTangent(), NoTangent()
    end

    return y, pool_sum_pullback
end

test_rrule(pool_sum, x, s1, s2);


using BenchmarkTools, Test

x = randn(100,100);
s1 = create_range(100, 3);
s2 = create_range(100, 2);

@benchmark gradient(a->sum(pool_native(a, s1, s2)), x)
@benchmark gradient(a->sum(pool_sum(a, s1, s2)), x)

@test gradient(a->sum(pool_native(a, s1, s2)), x) == gradient(a->sum(pool_sum(a, s1, s2)), x)


# Exercise 4: Segmented Hausdorff distance (Bonus)
function forward_pool_hausdorff(x::AbstractMatrix, seg₁::AbstractBags, seg₂::AbstractBags) 
    o = similar(x, (length(seg₁), length(seg₂)))
    indexes = Matrix{CartesianIndex{2}}(undef, length(seg₁), length(seg₂))
    for (i, sᵢ) in enumerate(seg₁)
        for (j, sⱼ) in enumerate(seg₂)
            min₁, argmin₁ = findmin(x[sᵢ, sⱼ], dims=1)
            min₂, argmin₂ = findmin(x[sᵢ, sⱼ], dims=2)
            max₁, argmax₁ = findmax(min₁)
            max₂, argmax₂ = findmax(min₂)
            argmaxmins = [argmin₁[argmax₁], argmin₂[argmax₂]]
            h, m = findmax([max₁, max₂])
            o[i, j] = h
            indexes[i, j] = argmaxmins[m]

        end
    end
    return o, indexes
end

function backward_pool_hausdorff(ȳ, x, seg₁, seg₂, argmaxmins)
    o = zero(x)
    for (i, sᵢ) in enumerate(seg₁)
        for (j, sⱼ) in enumerate(seg₂)
            segment = o[sᵢ, sⱼ] 
            segment[argmaxmins[i, j]] += ȳ[i, j]
            o[sᵢ, sⱼ] .= segment # updated segment 
        end
    end
    return NoTangent(), o, NoTangent(), NoTangent(), NoTangent()
end