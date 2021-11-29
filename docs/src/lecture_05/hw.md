# Homework 5: Root finding of polynomials
This homework should test your ability to use the knowledge of benchmarking, profiling and others to improve an existing implementation of root finding methods for polynomials. The provided [code](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/master/docs/src/lecture_05/root_finding.jl) is of questionable quality. In spite of the artificial nature, it should simulate a situation in which you may find yourself quite often, as it represents some intermediate step of going from a simple script to something, that starts to resemble a package.

## How to submit?
Put the modified `root_finding.jl` code inside `hw.jl`. Zip only this file (not its parent folder) and upload it to BRUTE. Your file should not use any dependency other than those already present in the `root_finding.jl`.

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (2 points)</header>
<div class="admonition-body">
```
Use profiler on the `find_root` function to find a piece of unnecessary code, that takes more time than the computation itself. The finding of roots with the polynomial 
```math
p(x) = (x - 3)(x - 2)(x - 1)x(x + 1)(x + 2)(x + 3) = x^7 - 14x^5 + 49x^3 - 36x
```
should not take more than `50μs` when running with the following parameters
```julia
atol = 1e-12
maxiter = 100
stepsize = 0.95

x₀ = find_root(p, Bisection(), -5.0, 5.0, maxiter, stepsize, atol)
x₀ = find_root(p, Newton(), -5.0, 5.0, maxiter, stepsize, atol)
x₀ = find_root(p, Secant(), -5.0, 5.0, maxiter, stepsize, atol)
```

Remove obvious type instabilities in both `find_root` and `step!` functions. Each variable with "inferred" type `::Any` in `@code_warntype` will be penalized.

**HINTS**:
- running the function repeatedly `1000x` helps in the profiler sampling
- focus on parts of the code that may have been used just for debugging purposes

```@raw html
</div></div>
<details class = "solution-body" hidden>
<summary class = "solution-header">Solution:</summary><p>
```

Nothing to see here.


```@raw html
</p></details>
```

# Voluntary exercise
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Voluntary exercise</header>
<div class="admonition-body">
```

Use `Plots.jl` to plot the polynomial $p$ on the interval $[-5, 5]$ and visualize the progress/convergence of each method, with a dotted vertical line and a dot on the x-axis for each subsequent root approximation `x̃`.

**HINTS**:
- plotting scalar function `f` - `plot(r, f)`, where `r` is a range of `x` values at which we evaluate `f`
- updating an existing plot - either `plot!(plt, ...)` or `plot!(...)`, in the former case the plot lives in variable `plt` whereas in the latter we modify some implicit global variable
- plotting dots - for example with `scatter`/`scatter!`
- `plot([(1.0,2.0), (1.0,3.0)], ls=:dot)` will create a dotted line from position `(x=1.0,y=2.0)` to `(x=1.0,y=3.0)`

```@raw html
</div></div>
```