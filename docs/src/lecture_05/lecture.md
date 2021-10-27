# [Benchmarking, profiling, and performance gotchas](@id perf_lecture)

This class is a short introduction to writing a performant code. As such, we want to cover
- how to identify weak spots in the code
- how to properly benchmark
- common performance anti-patterns
- Julia's "performance gotchas", by which we mean performance problems specific for Julia (typicall caused by the lack of understanding of Julia or by a errors in coversion from script to functions)

Though recall the most important rule of thumb: **Never optimize code from the very beggining.** A much more productive workflow is 
1. write the code that is idiomatic and easy to understand
2. cover the code with unit test, such that you know that the optimized code works the same as the original
3. optimize the code

Premature optimization have two drawbacks:
- you might end-up optimizing wrong thing, i.e. you will not optimize performance bottleneck, but something very different
- optimized code can be difficult to read and reason about, which means it is more difficult to make it right.

It frequently happens that Julia newbies asks on forum that their code in Julia is slow in comparison to the same code in Python (numpy). Most of the time, they make trivial mistakes and it is very educative to go over their mistakes
https://discourse.julialang.org/t/numpy-10x-faster-than-julia-what-am-i-doing-wrong-solved-julia-faster-now/29922

## Numpy 10x faster than julia what am i doing wrong? (solved julia faster now)
[adapted from](https://discourse.julialang.org/t/numpy-10x-faster-than-julia-what-am-i-doing-wrong-solved-julia-faster-now/29922
)

```julia
function f(p)														# line 1 
    t0,t1 = p 														# line 2
    m0 = [[cos(t0) - 1im*sin(t0)  0]; [0  cos(t0) + 1im*sin(t0)]]	# line 3
    m1 = [[cos(t1) - 1im*sin(t1)  0]; [0  cos(t1) + 1im*sin(t1)]]	# line 4
    r = m1*m0*[1. ; 0.] 											# line 5
    return abs(r[1])^2 												# line 6
end

function g(p,n)
    return [f(p[:,i]) for i=1:n]
end

g(rand(2,3),3)  # call to force jit compilation

n = 10^6
p = 2*pi*rand(2,n)

@elapsed g(p,n)

```

The first thing we do is to run Profiler, to identify, where the function spends most of the time.


## Detour: Julia's built-in profiler
Julia's built-in profiler is part of the standard library in the `Profile` module implementing a fairly standard sampling based profiler. It a nutshell it asks at regular intervals, where the code execution is currently and marks it and collects this information in some statistics. This allows us to analyze, where these "probes" have occured most of the time which implies those parts are those, where the execution of your function spends most of the time. As such, the profiler has two "controls", which is the `delay` between two consecutive probes and the maximum number of probes `n` (if the profile code takes a long time, you might need to increase it).
```
using Profile
Profile.init(; n = 989680, delay = 0.001))
@profile g(p,n)
Profile.clear()
@profile g(p,n)
```

The default `Profile.print` function shows the call-tree with count, how many times the probe occured in each function sorted from the most to least. The output is a little bit difficult to read and orrient in, therefore there are some visualization options.

What are our options?
- `ProfileView` is the workhorse with a GTK based api and therefore recommended for those with working GTK
- `ProfileSVG` is the `ProfileView` with the output exported in SVG format, which is viewed by most browser (it is also very convenient for sharing with others)
- `PProf.jl` is a front-end to Google's PProf profile viewer https://github.com/JuliaPerf/PProf.jl
- `StatProfilerHTML`  https://github.com/tkluck/StatProfilerHTML.jl
By personal opinion I mostly use ProfileView (or ProfileSVG) as it indicates places of potential type instability, which as will be seen later is very useful feature. 

Let's start with `ProfileSVG` and save the output to `profile.svg`
```
using ProfileSVG
ProfileSVG.save("/tmp/profile.svg")
```
which shows the output of the profiler as a flame graph which reads as follows:
- the width of the bar corresponds to time spent in the function
- red colored bars indicate type instabilities
- functions in bottom bars calles functions on top of upper bars
Function name contains location in files and particular line number called. GTK version is even "clickable" and opens the file in default editor.

## continueing with the example Numpy 10x...
We can see that the function is type stable and 2/3 of the time is spent in lines 3 and 4, which allocates arrays
```
[cos(t0) - 1im*sin(t0)  0; 
 0  cos(t0) + 1im*sin(t0)]
```
and
```
[cos(t1) - 1im*sin(t1)  0; 
 0  cos(t1) + 1im*sin(t1)]
```
.

Looking at function `f`, we see that in every call, it has to allocate arrays `m0` and `m1` on the heap. The allocation on heap is expensive, because it requires interaction with the operating system. Can we avoid it?
Repeated allocation can be frequently avoided by:
- preallocating arrays
- allocating objects on stack, which does not involve interacion with OS (but can be used in limited cases.)

Preallocation
```julia
function f!(m0, m1, p, u)   										# line 1 
    t0,t1 = p 														# line 2
    m0[1,1] = cos(t0) - 1im*sin(t0)									# line 3
    m0[2,2] = cos(t0) + 1im*sin(t0)									# line 4
    m1[1,1] = cos(t1) - 1im*sin(t1)									# line 5
    m1[2,2] = cos(t1) + 1im*sin(t1)									# line 6
    r = m1*m0*u 													# line 7
    return abs(r[1])^2 												# line 8
end

function g2(p,n)
	u = [1. ; 0.]
	m0 = [[cos(p[1]) - 1im*sin(p[1])  0]; [0  cos(p[1]) + 1im*sin(p[1])]]	# line 3
    m1 = [[cos(p[2]) - 1im*sin(p[2])  0]; [0  cos(p[2]) + 1im*sin(p[2])]]
    return [f!(m0, m1, p[:,i], u) for i=1:n]
end
```

## Detour no 2. Benchmarking

## Coming back from detour
```
using BenchmarkTools

julia> @benchmark g(p,n)
BenchmarkTools.Trial: 5 samples with 1 evaluation.
 Range (min … max):  1.168 s …   1.199 s  ┊ GC (min … max): 11.57% … 13.27%
 Time  (median):     1.188 s              ┊ GC (median):    11.91%
 Time  (mean ± σ):   1.183 s ± 13.708 ms  ┊ GC (mean ± σ):  12.10% ±  0.85%

  █ █                                 █ █                 █
  █▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  1.17 s         Histogram: frequency by time         1.2 s <

 Memory estimate: 1.57 GiB, allocs estimate: 23000002.
```

```
julia> @benchmark g2(p,n)
BenchmarkTools.Trial: 11 samples with 1 evaluation.
 Range (min … max):  413.167 ms … 764.393 ms  ┊ GC (min … max):  6.50% … 43.76%
 Time  (median):     426.728 ms               ┊ GC (median):     6.95%
 Time  (mean ± σ):   460.688 ms ± 102.776 ms  ┊ GC (mean ± σ):  12.85% ± 11.04%

  ▃█ █
  ██▇█▁▁▁▁▁▁▁▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇ ▁
  413 ms           Histogram: frequency by time          764 ms <

 Memory estimate: 450.14 MiB, allocs estimate: 4000021.

```

We can see that we have approximately 3-fold improvement.

Let's profile again and do not forget to use `Profile.clear()` to clear already stored probes.
```
Profile.clear()
@profile g2(p,n)
ProfileSVG.save("/tmp/profile2.svg")
```
What we can see the profiler now?
	- we spend a lot of time in `similar` in `matmul`, which is again an allocation of results for storing output of multiplication on line 7 matrix `r`.
	- the trigonometric operations on line 3-6 are very costly
	- Slicing `p` always allocates a new array and performs a deep copy.

Let's try to get rid of these.
```julia
using LinearAlgebra
@inline function initm!(m, t)
    st, ct = sincos(t) 
    @inbounds m[1,1] = Complex(ct, -st)								
    @inbounds m[2,2] = Complex(ct, st)   								
end

function f1!(r1, r2, m0, m1, t0, t1, u)   					
    initm!(m0, t0)
    initm!(m1, t1)
    mul!(r1, m0, u)
    mul!(r2, m1, r1)
    return @inbounds abs(@inbounds r2[1])^2
end

function g2(p,n)
	u = [1. ; 0.]
	m0 = [[cos(p[1]) - 1im*sin(p[1])  0]; [0  cos(p[1]) + 1im*sin(p[1])]]
    m1 = [[cos(p[2]) - 1im*sin(p[2])  0]; [0  cos(p[2]) + 1im*sin(p[2])]]
    r1 = m0*u
    r2 = m1*r1
    return [f1!(r1, r2, m0, m1, p[1,i], p[2,i], u) for i=1:n]
end
```

```julia
julia> @benchmark g2(p,n)
 Range (min … max):  193.922 ms … 200.234 ms  ┊ GC (min … max): 0.00% … 1.67%
 Time  (median):     195.335 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   196.003 ms ±   1.840 ms  ┊ GC (mean ± σ):  0.26% ± 0.61%

  █▁  ▁ ██▁█▁  ▁█ ▁ ▁ ▁        ▁   ▁     ▁   ▁▁   ▁  ▁        ▁
  ██▁▁█▁█████▁▁██▁█▁█▁█▁▁▁▁▁▁▁▁█▁▁▁█▁▁▁▁▁█▁▁▁██▁▁▁█▁▁█▁▁▁▁▁▁▁▁█ ▁
  194 ms           Histogram: frequency by time          200 ms <

 Memory estimate: 7.63 MiB, allocs estimate: 24.
 ```
Notice that now, we are about six times faster than the first solution, albeit passing the pre code is getting messy. Also notice that we spent a very little time in garbage collector. Running the profiler, we see that there is very little what we can do now. May-be, remove 
bounds checks and make the code a bit nicer.

Let's look at solution from a Discourse
```julia
using StaticArrays, BenchmarkTools

function f(t0,t1)
    cis0, cis1 = cis(t0), cis(t1)
    m0 = @SMatrix [ conj(cis0) 0 ; 0 cis0]
    m1 = @SMatrix [ conj(cis1) 0 ; 0 cis1]
    r = m1 * (m0 * @SVector [1. , 0.])
    return abs2(r[1])
end

g(p) = [f(p[1,i],p[2,i]) for i in axes(p,2)]
```

```
julia> @benchmark g(p)
 Range (min … max):  36.076 ms … 43.657 ms  ┊ GC (min … max): 0.00% … 9.96%
 Time  (median):     37.948 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   38.441 ms ±  1.834 ms  ┊ GC (mean ± σ):  1.55% ± 3.60%

        █▃▅   ▅▂  ▂
  ▅▇▇███████▅███▄████▅▅▅▅▄▇▅▇▄▇▄▁▄▇▄▄▅▁▄▁▄▁▄▅▅▁▁▅▁▁▅▄▅▄▁▁▁▁▁▅ ▄
  36.1 ms         Histogram: frequency by time        43.4 ms <

 Memory estimate: 7.63 MiB, allocs estimate: 2.
```
We can see that it is six-times faster than ours while also being much nicer to read and 
having almost no allocations. Where is the catch?
It uses `StaticArrays` which offers linear algebra primitices performant for vectors and matrices of small size. They are allocated on stack, therefore there is no pressure of GarbageCollector and the type is specialized on size of matrices (unlike regular matrices) works on arrays of an sizes. This allows the compiler to perform further optimizations like unrolling loops, etc.

What we have learned so far?
- Profiler is extremely useful in identifying functions, where your code spends most time.
- Memory allocation (on heap to be specific) can be very bad for the performance. We can generally avoided by pre-allocation (if possible) or allocating on the stack (Julia offers increasingly larger number of primitives for hits. We have already seen StaticArrays, DataFrames now offers for example String3, String7, String15, String31).
- Benchmarking is useful for comparison of solutions



The same function, but with keyword arguments, can be used to change these settings, however these settings are system dependent. For example on Windows, there is a known issue that does not allow to sample faster than at `0.003s` and even on Linux based system this may not do much. There are some further caveat specific to Julia:
- When running profile from REPL, it is usually dominated by the interactive part which spawns the task and waits for it's completion.
- Code has to be run before profiling in order to filter out all the type inference and interpretation stuff. (Unless compilation is what we want to profile.)
- When the execution time is short, the sampling may be insufficient -> run multiple times.

## Type stability
Sometimes it happens that we create a non-stable code, which might be difficult to spot at first, for a non-trained eye. A prototypical example of such bug is as follows
```julia
function poor_sum(x)
	s = 0
	n = length(x)
	for i in 1:n
		s += x[i]
	end
	s
end
```

```julia
x = randn(10^8);
julia> @benchmark poor_sum(x)
BenchmarkTools.Trial: 18 samples with 1 evaluation.
 Range (min … max):  246.936 ms … 325.800 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     300.235 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   292.046 ms ±  25.373 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▁   ▁ ▁        ▁   █        ▁    ▁     ▁  ▁   █ ▁▁ ▁ ▁      █
  █▁▁▁█▁█▁▁▁▁▁▁▁▁█▁▁▁█▁▁▁▁▁▁▁▁█▁▁▁▁█▁▁▁▁▁█▁▁█▁▁▁█▁██▁█▁█▁▁▁▁▁▁█ ▁
  247 ms           Histogram: frequency by time          326 ms <

 Memory estimate: 16 bytes, allocs estimate: 1.
 ```
 
Can we do better? Let's look what profiler says.
```
using Profile, ProfileSVG
Profile.clear()
@profile  poor_sum(x)
ProfileSVG.save("/tmp/profile4.svg")
```
THe profiler does not show any red, which means that according to the profilerthe code is type stable (and so does the `@code_typed poor_sum(x)` does not show anything bad.) Yet, we can see that the fourth line of the `poor_sum` function takes unusually long (there is a white area above, which means that the time spend in childs of that line (iteration and sum) does the sum to the time spent in the line, which is fishy). 

A close lookup on the code reveals that `s` is initialized as `Int64`, because `typeof(0)` is `Int64`. But then in the loop, we add to `s` a `Float64` because `x` is `Vector{Float64}`, which means during the execution, the type `s` changes the type.

So why nor compiler nor `@code_typed(poor_sum(x))` warns us about the type instability? This is because of the optimization called **small unions**, where Julia can optimize "small" type instabilitites (recall the second lecture).

We can fix it for example by initializing `x` to be the zero of an element type of the array `x` (though this solution technically assumes `x` is an array, which means that `poor_sum` will not work for generators)
```julia
function better_sum(x)
	s = zero(eltype(x))
	n = length(x)
	for i in 1:n
		s += x[i]
	end
	s
end
```

But there is no difference, due to small union optimization (the above would kill any performance in older versions.)
```julia
julia> @benchmark better_sum(x)
BenchmarkTools.Trial: 17 samples with 1 evaluation.
 Range (min … max):  171.921 ms … 344.261 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     308.825 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   294.903 ms ±  51.566 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

                           ▃                    ▃         ▃   █
  ▇▁▁▁▁▁▁▁▁▁▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▇▁▁▁▁▁▁▁▁▁▁▁▇▁█▁▇▇▁▁▁▁▇▁█▁▁▇█ ▁
  172 ms           Histogram: frequency by time          344 ms <

 Memory estimate: 16 bytes, allocs estimate: 1.```
```


The main problem with the above formulation is that Julia is checking that getting element of arrays from `x[i]` is within bounds. We can remove the check using `@inbounds` macro.
```julia
function fast_sum(x)
	n = length(x)
	n == 0 && return(zero(eltype(x)))
	n == 1 && return(x[1])
    @inbounds a1 = x[1]
    @inbounds a2 = x[2]
    v = a1 + a2
    for i = 2 : n
        @inbounds ai = x[i]
        v += ai
    end
    v
end
``` 

This gives us few orders of magnitude speed improvements. 
```
BenchmarkTools.Trial: 40 samples with 1 evaluation.
 Range (min … max):  122.005 ms … 135.203 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     127.112 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   127.306 ms ±   3.222 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▃      ▃   █    ▃█  ▃█▃            ▃
  ▇▇▁█▁▇▁▁▁▁█▁▇▇█▁▁▁▇██▇▁███▇▇▁▇▁▁▇▁▇▇▁▁█▁▇▁▇▁▇▁▁▁▁▁▁▁▁▁▁▁▇▇▁▁▇ ▁
  122 ms           Histogram: frequency by time          135 ms <

 Memory estimate: 16 bytes, allocs estimate: 1.
 ```


## Global variables introduces type instability
```julia
function implicit_sum()
	n = length(x)
	n == 0 && return(zero(eltype(x)))
	n == 1 && return(x[1])
    @inbounds a1 = x[1]
    @inbounds a2 = x[2]
    v = a1 + a2
    for i = 3 : n
        @inbounds ai = x[i]
        v += ai
    end
    v
end
x = randn(10^8);
```


```
julia> @benchmark implicit_sum()
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 10.837 s (11.34% GC) to evaluate,
 with a memory estimate of 8.94 GiB, over 499998980 allocations.
```
What? The same function where I made the parameters to be implicit has just turned **nine orders of magnitude** slower? 

Let's look what the debugger says
```julia
Profile.clear()
x = randn(10^4)
@profile implicit_sum()
ProfileSVG.save("/tmp/profile_5.svg")
```
which does not say anything except that there is a huge type-instability (red bar). In fact, the whole computation is dominated by Julia constantly determining the type (of something).

We have several tools that can help us to determine the type instability. 
- `@code_typed implicit_sum()`
- Cthulhu as `@descend implicit_sum()`
- JET (available with the nightly build of Julia and 1.7 pre-releases) `@report_opt implicit_sum()`

All of these tools tells us that the Julia's compiler cannot determine the type of `x`. But why? I can just invoke `typeof(x)` and I know immediately the type of `x`. 

To understand the problem, you have to think about the compiler.
1. You define function `implicit_sum().`
2. If you call `implicit_sum` and `x` has not exist, Julia will happily crash.
3. If you call `implicit_sum` and `x` exist, the function will gives you the result (albeit slowly). At this moment, Julia has to specialize `implicit_sum`. It has two options how to behave with respect to `x`. 
	a. She can assume that type of `x` is the current `typeof(x)` but that would mean that if a user redefines x and change the type, the specialization of the function `implicit_sum` will assume the wrong type of `x` and it can have unexpected results.
	b. She can take safe approach and determine the type of `x` inside the function `implicit_sum` and behave accordingly (recall that julia is dynamically type). Yet, not knowing the type precisely is absolute disaster for performance.

We can check tha `implicit_sum` gives correct version for other types
```julia
x = rand(Int, 1000)
implicit_sum() ≈ sum(x)
x = rand(UInt32, 1000)
implicit_sum() ≈ sum(x)
```

## Detour 3. JET
Abstract Interpreter JET

 of the childs does not match. Why is that? What we can see is Julia's optimization called **Small Unions** (we have talked about this in lecture on Type stability).


## return from Detour
```
using JET
@report_call sum("julia")

- performance tweaks
	+ deep-copy vs view
	+ boxing in closures
	+ memory layout matters
	+ bounds checking
	+ differnce between named tuple and dict
	+ effect of global variables
	+ IO

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