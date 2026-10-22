using BenchmarkTools
using LoopVectorization
using CairoMakie
using DelimitedFiles

# !!! RUN THIS WITH:  julia -t 4
RERUN = false

function f(x)
    @. 3*x^3 + 2*x^2 + x + 1
end

function f_turbo(x)
    @turbo @. 3*x^3 + 2*x^2 + x + 1
end

function f_tturbo(x)
    @tturbo @. 3*x^3 + 2*x^2 + x + 1
end

sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
funcs = [f, f_turbo, f_tturbo]

if RERUN
    times = map(funcs) do fn
        ts = map(sizes) do n
            x = rand(n)
            r = @benchmark $fn($x)
            t = minimum(r.times) ./ 10^3
            @info "Benchmarking" fn n t
            t
        end
        writedlm("julia-$(nameof(fn)).txt", ts)
        nameof(fn) => ts
    end |> Dict
else
    times = Dict(nameof(fn)=>readdlm("julia-$(nameof(fn)).txt", Float64)|>vec for fn in funcs)
end

fig = Figure()
ax = Axis(
    fig[1,1],
    title=L"f(x) = 3x^3 + 2x^2 + x + 1",
    ylabel=L"\mu s", xlabel=L"$n$ where $[x = \text{rand}(n)]$",
    xticks=sizes, yticks=[0.1,1,10,100,1000,10000],
    xscale=log10, yscale=log10,
)

lines!(ax, sizes, times[:f], linewidth=2)
scatter!(ax, sizes, times[:f], label="Julia", marker=:circle)

lines!(ax, sizes, times[:f_turbo], linewidth=2)
scatter!(ax, sizes, times[:f_turbo], label="@turbo", marker=:rect)

lines!(ax, sizes, times[:f_tturbo], linewidth=2)
scatter!(ax, sizes, times[:f_tturbo], label="@tturbo", marker=:star5)

lines!(ax, sizes, readdlm("numpy.txt", Float64)|>vec, linewidth=2)
scatter!(ax, sizes, readdlm("numpy.txt", Float64)|>vec, label="Numpy", marker=:diamond)

lines!(ax, sizes, readdlm("jax.txt", Float64)|>vec, linewidth=2)
scatter!(ax, sizes, readdlm("jax.txt", Float64)|>vec, label="JAX", marker=:hexagon)

axislegend(position=:lt)
save("bench.png", fig)
