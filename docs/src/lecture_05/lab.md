# Lab 05: Benchmarking, profiling and performance gotchas
Performance is crucial in scientific computing. There is a big difference if your experiments run one minute or one hour. We have already developed quite a bit of code, both in packages and independent, on which we are going to present some of the tooling that Julia provides to try find performance bottlenecks. Performance of your code or more precisely the speed of execution is of course relative (preference, expectation, existing code) and it's hard to find the exact threshold when we should start to care about it. When starting out with Julia, we recommend not to get bogged down by the performance side of things straightaway, but just design the code in the way that feels natural to you. As opposed to other languages Julia offers you to write the things "like you are used" (depending on your background), e.g. for cycles are as fast as in C; vectorization of mathematical operators works the same or even better than in MATLAB, NumPy. 


Once you have tested the functionality, you can start exploring the performance of your code by different means:
- manual code inspection - identifying performance gotchas (tedious, requires skill) *specific cases should be covered in the lecture*
- automatic code inspection - *can we cover the JET.jl*? (not as powerful as in static typed languages?)
- benchmarking - measuring variability in execution time, comparing with some baseline (only a statistic, non-specific)
- profiling - sampling the execution at regular intervals to obtain time spent at different sections (no parallelism, ...)
- allocation tracking - similar to profiling but specifically looking at allocations (only one side of the story)
- 

## Checking type stability
Recall that type stable function is written in a way, that allows Julia's compiler to infer all the types of all the variables and produce an efficient native code implementation without the need of boxing some variables in a structure whose types is known only during runtime. Probably unbeknown to you we have already seen an example of type unstable function (at least in some situations) in the first lab, where we have defined the `polynomial` function:

```@repl lab05_polynomial
function polynomial(a, x)
    accumulator = 0
    for i in length(a):-1:1
        accumulator += x^(i-1) * a[i] # ! 1-based indexing for arrays
    end
    return accumulator
end
```

The exact form of compiled code and also the type stability depends on the arguments of the function. Let's explore the following two examples of calling the function
a. 
```@repl lab05_polynomial
a = [-19, 7, -4, 6]
x = 3
polynomial(a, x)
```

b.
```@repl lab05_polynomial
af = Float64.(a)
xf = 3.0
polynomial(af, xf)
```

The result they produce is the same numerically, however it differs in the output type. Though you have probably not noticed it, there should also be a significant difference in runtime (assuming that you have run it once more after its compilation) (*CONFIRM*). It is probably a surprise to no one that one of the function that were compiled is type unstable. This can be check with the `@code_warntype` macro:
```@repl lab05_polynomial
using InteractiveUtils
@code_warntype polynomial(a, x)  # type stable
@code_warntype polynomial(af, x) # type unstable
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Create a new function `polynomial_stable`, which is type stable and measure the difference in evaluation time.
- may require a longer polynomial (i.e. longer for loop)
- compile it before running `@time`


```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```



```@raw html
</p></details>
```

## Benchmarking



## Profiling


```@repl lab05_polynomial
using Profile, ProfileSVG
using BenchmarkTools

a = rand(-20:20, 1000); # slightly longer polynomial
# for i in 1:length(a)
#     @btime polynomial($(a[1:i]), $x)
# end

@profview polynomial(a, x)
```

```@repl lab05_polynomial
function polynomial(a, x::AbstractMatrix)
    accumulator = zeros(eltype(x), size(x))
    for i in length(a):-1:1
        accumulator += x^(i-1) * a[i]
    end
    return accumulator
end

a = rand(-20:20, 100);
A = rand(100,100)

@profview polynomial(a, A)
ProfileSVG.save("./prof.svg")
```



```@repl lab05_polynomial
run_polynom = () -> begin for _ in 1:10000 
```

## Ecosystem debugging



- ideally I should import the `Ecosystem` pkg created in previous lab 4
```julia
using Profile, ProfileSVG
using EcosystemCore

n_grass       = 500
regrowth_time = 17.0

n_sheep         = 100
Δenergy_sheep   = 5.0
sheep_reproduce = 0.5
sheep_foodprob  = 0.4

n_wolves       = 8
Δenergy_wolf   = 17.0
wolf_reproduce = 0.03
wolf_foodprob  = 0.02

gs = [Grass(id, regrowth_time) for id in 1:n_grass];
ss = [Sheep(id, 2*Δenergy_sheep, Δenergy_sheep, sheep_reproduce, sheep_foodprob) for id in n_grass+1:n_grass+n_sheep];
ws = [Wolf(id, 2*Δenergy_wolf, Δenergy_wolf, wolf_reproduce, wolf_foodprob) for id in n_grass+n_sheep+1:n_grass+n_sheep+n_wolves];
world = World(vcat(gs, ss, ws));

# precompile everything
simulate!(w, 1, [w -> @info agent_count(w)])

@profview simulate!(w, 100, [w -> @info agent_count(w)])
ProfileSVG.save("./ecosystem.svg")
```

- investigating red bars in the top

```julia
using Random
using EcosystemCore.StatsBase

# original
function EcosystemCore.find_rand(f, w::World)
    xs = filter(f, w.agents |> values |> collect)
    isempty(xs) ? nothing : sample(xs)
end

# Vasek
function EcosystemCore.find_rand(f, w::World)
    ks = collect(keys(w.agents))
    for i in randperm(length(w.agents))
        a = w.agents[ks[i]]
        f(a) && return a
    end
    return nothing
end

# mine
function EcosystemCore.find_rand(f, w::World)
    for i in shuffle!(collect(keys(w.agents)))
        a = w.agents[i]
        f(a) && return a
    end
    return nothing
end
```

- what about code stability
    + original version is unstable
    + fixed version infers return type as Union{Nothing, Agent}

```julia
w = ws[1] # I just need an instance to get the correct dispatch on eats
f = x -> EcosystemCore.eats(w, x)
EcosystemCore.find_rand(f, world)
@code_warntype EcosystemCore.find_rand(f, world)
```

```julia
@code_warntype EcosystemCore.find_food(w, world) # unstable as expected
@code_warntype EcosystemCore.find_mate(w, world) # unstable as expected

# what does the result of an agent step looks like?
@code_warntype EcosystemCore.agent_step!(w, world)

@code_warntype EcosystemCore.world_step!(world)
```

- does it have any performance penalties
```julia
using BenchmarkTools

gs = [Grass(id, regrowth_time) for id in 1:n_grass];
ss = [Sheep(id, 2*Δenergy_sheep, Δenergy_sheep, sheep_reproduce, sheep_foodprob) for id in n_grass+1:n_grass+n_sheep];
ws = [Wolf(id, 2*Δenergy_wolf, Δenergy_wolf, wolf_reproduce, wolf_foodprob) for id in n_grass+n_sheep+1:n_grass+n_sheep+n_wolves];
world = World(vcat(gs, ss, ws));

# does not work with the setup (works only with one setup variable)
@benchmark begin
    Random.seed!(7); 
    simulate!(World(vcat(g, s, w)), 10) 
end setup=(g = copy(gs), s = copy(ss), w = copy(ws))


# begin and end has to be used
# also deepcopy is needed
@benchmark begin
    Random.seed!(7); 
    simulate!(World(vcat(g, s, w)), 10) 
end setup=begin g, s, w = deepcopy(gs), deepcopy(ss), deepcopy(ws) end


@benchmark simulate!(World(vcat(g, s, w)), 10) setup=(g=copy(gs))

```