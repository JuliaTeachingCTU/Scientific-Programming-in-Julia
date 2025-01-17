using Pkg
Pkg.activate(@__DIR__)
using GLMakie
using BenchmarkTools
using Distributed
using SharedArrays

function juliaset_pixel(z₀, c)
    z = z₀
    for i in 1:255
        abs2(z)> 4.0 && return (i - 1)%UInt8
        z = z*z + c
    end
    return UInt8(255)
end

function juliaset_column!(img, c, n, colj, j)
    x = -2.0 + (j-1)*4.0/(n-1)
    for i in 1:n
        y = -2.0 + (i-1)*4.0/(n-1)
        @inbounds img[i, colj] = juliaset_pixel(x+im*y, c)
    end
    nothing
end

function juliaset_columns(c, n, columns)
    img = Array{UInt8,2}(undef, n, length(columns))
    for (colj, j) in enumerate(columns)
        juliaset_column!(img, c, n, colj, j)
    end
    img
end

function juliaset_distributed(x, y, partitions = nworkers(), n = 1000)
    c = x + y*im
    columns = Iterators.partition(1:n, div(n, partitions))
    slices = pmap(cols -> juliaset_columns(c, n, cols), columns)
    reduce(hcat, slices)
end

# @btime juliaset_distributed(-0.79, 0.15)

# frac = juliaset_distributed(-0.79, 0.15)
# plot(heatmap(1:size(frac,1), 1:size(frac,2), frac, color=:Spectral))


####
#   Let's work out the shared array approach
####
function juliaset_column!(img, c, n, j)
    x = -2.0 + (j-1)*4.0/(n-1)
    for i in 1:n
        y = -2.0 + (i-1)*4.0/(n-1)
        @inbounds img[i, j] = juliaset_pixel(x+im*y, c)
    end
    nothing
end

function juliaset_range!(img, c, n, columns)
    for j in columns
        juliaset_column!(img, c, n, j)
    end
    nothing
end

function juliaset_shared(x, y, partitions = nworkers(), n = 1000)
    c = x + y*im
    columns = Iterators.partition(1:n, div(n, partitions))
    img = SharedArray{UInt8,2}((n, n))
    slices = pmap(cols -> juliaset_range!(img, c, n, cols), columns)
    img
end

# juliaset_shared(-0.79, 0.15)
# juliaset_shared(-0.79, 0.15, 16)
