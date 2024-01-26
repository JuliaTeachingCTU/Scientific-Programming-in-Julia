using BenchmarkTools

function polynomial(a, x)
    accumulator = 0
    for i in length(a):-1:1
        accumulator += x^(i-1) * a[i] # ! 1-based indexing for arrays
    end
    return accumulator
end

function polynomial_stable(a, x)
    accumulator = zero(x)
    for i in length(a):-1:1
        accumulator += x^(i-1) * a[i]
    end
    accumulator
end

function run_polynomial_stable(a, x, n)
    for _ in 1:n
        polynomial_stable(a, x)
    end
end

function run_polynomial(a, x, n)
    for _ in 1:n
        polynomial(a, x)
    end
end

function polynomial_horner(a, x)
    accumulator = a[end] * one(x)
    for i in length(a)-1:-1:1
        accumulator = accumulator * x + a[i]
    end
    accumulator
end
function run_polynomial_horner(a, x, n)
    for _ in 1:n
        polynomial_horner(a, x)
    end
end



a = rand(-10:10, 1000) # using longer polynomial
xf = 3.0


run_polynomial(a, xf, Int(1e5))
@profview run_polynomial(a, xf, Int(1e5))

run_polynomial_stable(a, xf, Int(1e5))
@profview run_polynomial_stable(a, xf, Int(1e5))

run_polynomial_horner(a, xf, Int(1e5))
@profview run_polynomial_horner(a, xf, Int(1e5))

# a = rand(-10:10, 100)
# polynomial(a,3)
# @profview for _ in 1:100000 polynomial(a, 3) end
# 
# polynomial(a,3.0)
# #@profview for _ in 1:100000 polynomial(a, 3.0) end
# @btime for _ in 1:100000 polynomial($a, 3.0) end
# 
# # @time for _ in 1:1000 polynomial(a, 3.0) end
# 
