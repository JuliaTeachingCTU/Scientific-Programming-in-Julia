# Parallel programming with Julia
Julia offers different levels of parallel programming
- distributed processing, where jobs are split among different Julia processes
- multi-threadding, where jobs are split among multiple threads within the same processes
- SIMD instructions
- Task switching.

In this lecture, we will focus mainly on the first two, since SIMD instructions are mainly used for optimization of loops, and task switching is not a true paralelism, but allows to run a different task when one task is waiting for example for IO.



## Multi-Threadding 
An example adapted from [Eric Aubanel](http://www.cs.unb.ca/~aubanel/JuliaMultithreadingNotes.html).

For ilustration, we will use Julia set fractals, ad they can be easily paralelized. Some fractals (Julia set, Mandelbrot) are determined by properties of some complex-valued functions. Julia set counts, how many iteration is required for  ``f(z)=z^2+c`` to be bigger than two in absolute value, ``|f(z)>=2``. The number of iterations can then be mapped to the pixel's color, which creates a nice visualization we know.
```julia
function juliaset_pixel(z₀, c)
    z = z₀
    for i in 1:255
        abs2(z)> 4.0 && return (i - 1)%UInt8
        z = z*z + c
    end
    return UInt8(255)
end
```

In our multi-threadding experiments, the level of granulity will be one column, since calculation of single pixel is so fast, that thread-switching will have much higher overhead.
```julia
function juliaset_column!(img, c, n, j)
    x = -2.0 + (j-1)*4.0/(n-1)
    for i in 1:n
        y = -2.0 + (i-1)*4.0/(n-1)
        @inbounds img[i,j] = juliaset_pixel(x+im*y, c)
    end
    nothing
end
```

To calculate full image
```julia
function juliaset_image!(img, c, n)
    for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end
```

```julia
function juliaset(x, y, n=1000, method = juliaset_image!, extra...)
    c = x + y*im
    img = Array{UInt8,2}(undef,n,n)
    method(img, c, n, extra...)
    return img
end
```

and run it
```julia
using Plots
frac = juliaset(-0.79, 0.15)
plot(heatmap(1:size(frac,1),1:size(frac,2), frac, color=:Spectral))
```

To observe the execution length, we will use `BenchmarkTools.jl` again
```
using BenchmarkTools
julia> @btime juliaset(-0.79, 0.15);
  39.822 ms (2 allocations: 976.70 KiB)
```

Let's now try to speed-up the calculation using multi-threadding. `Julia v0.5` has introduced multi-threadding with static-scheduller with a simple syntax: just prepend the for-loop with a `Threads.@threads` macro. With that, the first multi-threaded version will looks like
```julia
function juliaset_static!(img, c, n)
    Threads.@threads for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end
```
with benchmark
```
julia> @btime juliaset(-0.79, 0.15, 1000, juliaset_static!);
  16.206 ms (26 allocations: 978.42 KiB)
```
Although we have used four-threads, the speed improvement is ``2.4``. Why is that? The reason is that the static scheduller partition the total number of columns (1000) into equal parts, where the total number of parts is equal to the number of threads, and assign each to a single thread. In our case, we will have four parts each of size 250. Since the execution time of computing value of each pixel is not the same and therefore there will be threads who will not do anything. This problem is addressed by dynamic schedulling, which divides the problem into smaller parts, and when a thread is finished with one part, it assigned new not-yet computed part.

Dynamic scheduller is supported using `Threads.@spawn` macro. The prototypical approach is the fork-join model, where one recursivelly partitions the problems and wait in each thread for the other
```julia
function juliaset_recspawn!(img, c, n, lo=1, hi=n, ntasks=128)
    if hi - lo > n/ntasks-1
        mid = (lo+hi)>>>1
        finish = Threads.@spawn juliaset_recspawn!(img, c, n, lo, mid, ntasks)
        juliaset_recspawn!(img, c, n, mid+1, hi, ntasks)
        wait(finish)
        return
    end
    for j in lo:hi
        juliaset_column!(img, c, n, j)
    end
    nothing
end
```
Measuring the time we observe four-times speedup, which corresponds to the number of threads.
```julia
julia> @btime juliaset(-0.79, 0.15, 1000, juliaset_recspawn!);
  10.326 ms (142 allocations: 986.83 KiB)
```
Due to task switching overhead, increasing the granularity might not pay off.
```julia
4 tasks: 16.262 ms (21 allocations: 978.05 KiB)
8 tasks: 10.660 ms (45 allocations: 979.80 KiB)
16 tasks: 10.326 ms (142 allocations: 986.83 KiB)
32 tasks: 10.786 ms (238 allocations: 993.83 KiB)
64 tasks: 10.211 ms (624 allocations: 1021.89 KiB)
128 tasks: 10.224 ms (1391 allocations: 1.05 MiB)
256 tasks: 10.617 ms (2927 allocations: 1.16 MiB)
512 tasks: 11.012 ms (5999 allocations: 1.38 MiB)
```

```julia
using FLoops, FoldsThreads
function juliaset_folds!(img, c, n)
    @floop ThreadedEx(basesize = 2) for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end
julia> @btime juliaset(-0.79, 0.15, 1000, juliaset_folds!);
  10.575 ms (52 allocations: 980.12 KiB)
```
where `basesize` is the size of the part, in this case 2 columns.
```julia
function juliaset_folds!(img, c, n)
    @floop WorkStealingEx(basesize = 2) for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end
julia> @btime juliaset(-0.79, 0.15, 1000, juliaset_folds!);
  10.575 ms (52 allocations: 980.12 KiB)
```

```julia
function juliaset_folds!(img, c, n)
    @floop DepthFirstEx(basesize = 2) for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end
julia> @btime juliaset(-0.79, 0.15, 1000, juliaset_folds!);
  10.421 ms (3582 allocations: 1.20 MiB)
```