# [Benchmarking, profiling, and performance gotchas](@id perf_lecture)
- write the code well from the start -> unit test -> start thinking about performance
	+ danger of premature optimization
- human intuition is bad when reasoning about where the program spends most of the time
	+ how to identify them <- profiler (intro to sampling based profilers)
- performance tweaks
	+ effect of global variables
	+ memory allocations matters (mainly on heap - theoretical intro to this topic)
	+ differnce between named tuple and dict
	+ boxing in closures
	+ IO
	+ memory layout matters

- create a list of examples for the lecture


## Avoid non-const globals
- modified https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-global-variables
```julia
using BenchmarkTools

x = rand(1000)

function loop_over_global()
    s = 0.0
    for i in x
        s += i
    end
    return s
end

@btime loop_over_global()
```
- can be alleviated with type annotation, but is not recommended
```julia
function loop_over_global()
    s = 0.0
    for i in x::Vector{Float64}
        s += i
    end
    return s
end
@btime loop_over_global()
```

## Avoid changing of type of variables
- https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-changing-the-type-of-a-variable
```julia
function foo()
    x = 1
    for i = 1:10
        x /= rand()
    end
    return x
end
@btime foo()
```
- produces the same output as the following, but this is now type stable and a bit faster
```julia
function foo()
    x = 1.0
    for i = 1:10
        x /= rand()
    end
    return x
end
@btime foo()
```

## Memory allocations matter
- https://docs.julialang.org/en/v1/manual/performance-tips/#Pre-allocating-outputs
- `xinc` allocates a new array each time it is called (`1e7` allocations)
```julia
function xinc(x)
    return [x, x+1, x+2]
end;

function loopinc()
    y = 0
    for i = 1:10^7
        ret = xinc(i)
        y += ret[2]
    end
    return y
end;
@btime loopinc()
```
- `xinc!` now only modifies the input vector (`1` allocation)
```julia
function xinc!(ret::AbstractVector{T}, x::T) where T
	ret[1] = x
	ret[2] = x+1
	ret[3] = x+2
	nothing
end;

function loopinc_prealloc()
	ret = Vector{Int}(undef, 3)
	y = 0
	for i = 1:10^7
	    xinc!(ret, i)
	    y += ret[2]
	end
	return y
end;
@btime loopinc_prealloc()
```
- `StaticArrays.jl` to the rescue if the arrays are small and of known size

## Memory layout matters
- https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-column-major I don't like that example much
- computing norm over columns is faster than over rows (adapted from https://stackoverflow.com/questions/65237635/transpose-matrix-and-keep-column-major-memory-layout)
```julia
using LinearAlgebra, BenchmarkTools
A = rand(10000, 10000);

@btime mapslices(norm, $A, dims=1) # columns
@btime mapslices(norm, $A, dims=2) # rows
```

## Using named tuple instead of dict
```julia
params_dict = Dict(:stepsize => 0.01, :h => 0.001, :iters => 500)
params_tuple = (;stepsize = 0.01, h=0.001, iters=500)

function find_min!(f, x, p)
	for i in 1:p[:iters]
		x̃ = x + p[:h]
		fx = f(x)
		x -= p[:stepsize] * (f(x̃) - fx)/p[:h]
	end
	x
end

x₀ = rand()
f = x -> x^2
find_min!(f, x₀, params_tuple)
@btime find_min!($f, $x₀, $params_dict)
@btime find_min!($f, $x₀, $params_tuple)
```

## Performance of captured variable
- inspired by https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
- an example of closure seen in previous lectures, reference `x` has to be included 
```julia
x = rand(1000)

function adder(shift)
    return y -> shift + y
end

function adder_typed(shift::Float64)
    return y -> shift + y
end

function adder_let(shift::Float64)
	f = let shift=shift
		y -> shift + y
    end
    return f
end

f = adder(3.0)
ft = adder_typed(3.0)
fl = adder_let(3.0)

@btime f.($x);
@btime ft.($x);
@btime fl.($x);
@btime $x .+ 3.0;

```
- cannot get the same performance as native call, might be affected by broadcasting (?)
- `fl` should attain the same performance as native call


## Don't use IO unless you have to
- debug printing in performance critical code should be kept to minimum or using in memory/file based logger in stdlib `Logging.jl`
```julia
function find_min!(f, x, p; verbose=true)
	for i in 1:p[:iters]
		x̃ = x + p[:h]
		fx = f(x)
		x -= p[:stepsize] * (f(x̃) - fx)/p[:h]
		verbose && println("x = ", x, " | f(x) = ", fx)
	end
	x
end

@btime find_min!($f, $x₀, $params_tuple; verbose=true)
@btime find_min!($f, $x₀, $params_tuple; verbose=false)
```
- interpolation of strings is even worse https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-string-interpolation-for-I/O
```julia
function find_min!(f, x, p; verbose=true)
	for i in 1:p[:iters]
		x̃ = x + p[:h]
		fx = f(x)
		x -= p[:stepsize] * (f(x̃) - fx)/p[:h]
		verbose && println("x = $x | f(x) = $fx")
	end
	x
end
@btime find_min!($f, $x₀, $params_tuple; verbose=true)
```