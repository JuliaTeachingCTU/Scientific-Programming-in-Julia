using BenchmarkTools
using LoopVectorization

function f(x)
    @. 3*x^3 + 2*x^2 + x + 1
end

function ft(x)
    @turbo @. 3*x^3 + 2*x^2 + x + 1
end

function g(x)
    @tturbo @. 3*x^3 + 2*x^2 + x + 1
end

sizes = [10,100,1000,10000, 100000, 1000000, 10000000]
j_ts = map(sizes) do n
    x = rand(n)
    r = @benchmark f($x)
    display(r)
    minimum(r.times) ./ 10^3
end

t_ts = map(sizes) do n
    x = rand(n)
    r = @benchmark ft($x)
    display(r)
    minimum(r.times) ./ 10^3
end

tt_ts = map(sizes) do n
    x = rand(n)
    r = @benchmark g($x)
    display(r)
    minimum(r.times) ./ 10^3
end


using Plots, DelimitedFiles
p1 = plot(
    sizes, j_ts,
    label="Julia", yscale=:log10, xscale=:log10,
    title="f(x) = 3x³ + 2x² + x + 1",
    ylabel="μs", xlabel="x-size",
    xticks=sizes, yticks=[0.1,1,10,100,1000,10000],
    lw=2, legend=:topleft
)
plot!(p1, sizes, t_ts, label="@turbo")
plot!(p1, sizes, tt_ts, label="@tturbo")
plot!(p1, sizes, readdlm("numpy.txt", Float64), label="Numpy")
plot!(p1, sizes, readdlm("jax.txt", Float64), label="JAX")
savefig(p1, "bench.png")
