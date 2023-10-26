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


using CairoMakie, DelimitedFiles
fig = Figure()
ax = Axis(
    fig[1,1], title="f(x) = 3x³ + 2x² + x + 1",
    xticks=sizes, yticks=[0.1,1,10,100,1000,10000],
    xscale=log10, yscale=log10,
    ylabel="μs", xlabel="x [x=rand(n)]"
)
lines!(ax, sizes, j_ts, linewidth=2)
scatter!(ax, sizes, j_ts, label="Julia", marker=:circle)
lines!(ax, sizes, t_ts, linewidth=2)
scatter!(ax, sizes, t_ts, label="@turbo", marker=:rect)
lines!(ax, sizes, tt_ts, linewidth=2)
scatter!(ax, sizes, tt_ts, label="@tturbo", marker=:star5)
lines!(ax, sizes, readdlm("numpy.txt", Float64)|>vec, linewidth=2)
scatter!(ax, sizes, readdlm("numpy.txt", Float64)|>vec, label="Numpy", marker=:diamond)
lines!(ax, sizes, readdlm("jax.txt", Float64)|>vec, linewidth=2)
scatter!(ax, sizes, readdlm("jax.txt", Float64)|>vec, label="JAX", marker=:hexagon)
axislegend(position=:lt)
save("bench.png", fig)
