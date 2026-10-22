using LinearAlgebra
using Printf

function _polynomial(a, x)
    accumulator = a[end] * one(x)
    for i in length(a)-1:-1:1
        accumulator = accumulator * x + a[i]
    end
    accumulator  
end

# definition of polynom
struct Polynom{C}
    coefficients::C
    Polynom(coefficients::CC) where CC = coefficients[end] == 0 ? throw(ArgumentError("Coefficient of the highest exponent cannot be zero.")) : new{CC}(coefficients)
end

# based on https://github.com/JuliaMath/Polynomials.jl
function from_roots(roots::AbstractVector{T}; aₙ = one(T)) where {T}
    n = length(roots)
    c = zeros(T, n+1)
    c[1] = one(T)
    for j = 1:n
        for i = j:-1:1
            c[i+1] = c[i+1]-roots[j]*c[i]
        end
    end
    return Polynom(aₙ.*reverse(c))
end

(p::Polynom)(x) = _polynomial(p.coefficients, x)
degree(p::Polynom) = length(p.coefficients) - 1

function _derivativeof(p::Polynom)
    n = degree(p)
    n > 1 ? Polynom([(i - 1)*p.coefficients[i] for i in 2:n+1]) : error("Low degree of a polynomial.")
end
LinearAlgebra.adjoint(p::Polynom) = _derivativeof(p)

function Base.show(io::IO, p::Polynom)
    n = degree(p)
    a = reverse(p.coefficients)
    for (i, c) in enumerate(a[1:end-1])
        if (c != 0)
            c < 0 && print(io, " - ")
            c > 0 && i > 1 && print(io, " + ")
            print(io, "$(abs(c))x^$(n - i + 1)")
        end
    end
    c = a[end]
    c > 0 && print(io, " + $(c)")
    c < 0 && print(io, " - $(abs(c))")
end

# default optimization parameters
atol = 1e-12
maxiter = 100
stepsize = 0.95

# definition of optimization methods
abstract type RootFindingMethod end
struct Newton <: RootFindingMethod end
struct Secant <: RootFindingMethod end
struct Bisection <: RootFindingMethod end

init!(::Bisection, p, a, b) = sign(p(a)) != sign(p(b)) ? (a, b) : throw(ArgumentError("Signs at both ends are the same."))
init!(::RootFindingMethod, p, a, b) = (a, b)

function step!(::Newton, poly::Polynom, xᵢ, step_size)
    _, x̃ = xᵢ
    dp = p'
    x̃, x̃ - step_size*p(x̃)/dp(x̃)
end

function step!(::Secant, poly::Polynom, xᵢ, step_size)
    x, x̃ = xᵢ
    dpx = (p(x) - p(x̃))/(x - x̃)
    x̃, x̃ - stepsize*p(x̃)/dpx
end

function step!(::Bisection, poly::Polynom, xᵢ, step_size)
    x, x̃ = xᵢ
    midpoint = (x + x̃)/2
    if sign(p(midpoint)) == sign(p(x̃))
        x̃ = midpoint
    else 
        x = midpoint
    end
    x, x̃
end

function find_root(p::Polynom, rfm=Newton, a=-5.0, b=5.0, max_iter=100, step_size=0.95, tol=1e-12)
    x, x̃ = init!(rfm, p, a, b)
    for i in 1:maxiter
        x, x̃ = step!(rfm, p, (x, x̃), step_size)
        val = p(x̃)
        @printf "x = %.5f | x̃ = %.5f | p(x̃) = %g\n" x x̃ val
        abs(val) < atol && return x̃
    end
    println("Method did not converge in $(max_iter) iterations to a root within $(tol) tolerance.")
    return x̃
end

# test code 
poly = Polynom(rand(4))
p = from_roots([-3, -2, -1, 0, 1, 2, 3])
dp = p'
p(3.0), dp(3.0)

x₀ = find_root(p, Bisection(), -5.0, 5.0)
x₀ = find_root(p, Newton(), -5.0, 5.0)
x₀ = find_root(p, Secant(), -5.0, 5.0)
