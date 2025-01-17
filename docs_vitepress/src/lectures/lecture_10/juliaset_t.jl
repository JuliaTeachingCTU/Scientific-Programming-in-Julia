using Plots, BenchmarkTools
Threads.nthreads()
function juliaset_pixel(z₀, c)
    z = z₀
    for i in 1:255
        abs2(z)> 4.0 && return (i - 1)%UInt8
        z = z*z + c
    end
    return UInt8(255)
end

function juliaset_column!(img, c, n, j)
    x = -2.0 + (j-1)*4.0/(n-1)
    for i in 1:n
        y = -2.0 + (i-1)*4.0/(n-1)
        @inbounds img[i,j] = juliaset_pixel(x+im*y, c)
    end
    nothing
end

function juliaset_single!(img, c, n)
    for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end

function juliaset(x, y, n=1000, method = juliaset_single!, extra...)
    c = x + y*im
    img = Array{UInt8,2}(undef,n,n)
    method(img, c, n, extra...)
    return img
end

# frac = juliaset(-0.79, 0.15)
# plot(heatmap(1:size(frac,1),1:size(frac,2), frac, color=:Spectral))


@btime juliaset(-0.79, 0.15);

function juliaset_static!(img, c, n)
    Threads.@threads for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end

@btime juliaset(-0.79, 0.15, 1000, juliaset_static!);


using Folds
function juliaset_folds!(img, c, n)
    Folds.foreach(j -> juliaset_column!(img, c, n, j), 1:n)
    nothing
end
julia> @btime juliaset(-0.79, 0.15, 1000, juliaset_folds!);
  16.267 ms (25 allocations: 978.20 KiB)

function juliaset_folds!(img, c, n, nt)
    parts = collect(Iterators.partition(1:n, cld(n, nt)))
    Folds.foreach(parts) do ii
        foreach(j ->juliaset_column!(img, c, n, j), ii)
    end
    nothing
end
julia> @btime juliaset(-0.79, 0.15, 1000, (args...) -> juliaset_folds!(args..., 16));
  16.716 ms (25 allocations: 978.61 KiB)


using FLoops, FoldsThreads
function juliaset_folds!(img, c, n)
    @floop ThreadedEx(basesize = 2) for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end
@btime juliaset(-0.79, 0.15, 1000, juliaset_folds!);