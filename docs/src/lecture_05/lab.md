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

The exact form of compiled code and also the type stability depends on the arguments of the function. Let's explore the following two examples of calling the function:

- Integer number valued arguments
```@example lab05_polynomial
a = [-19, 7, -4, 6]
x = 3
polynomial(a, x)
```
    
- Float number valued arguments
```@example lab05_polynomial
af = Float64.(a)
xf = 3.0
polynomial(af, xf)
```

The result they produce is the "same" numerically, however it differs in the output type. Though you have probably not noticed it, there should be a difference in runtime (assuming that you have run it once more after its compilation). It is probably a surprise to no one, that one of the function that has been compiled is type unstable. This can be check with the `@code_warntype` macro:
```@repl lab05_polynomial
using InteractiveUtils #hide
@code_warntype polynomial(a, x)  # type stable
@code_warntype polynomial(af, xf) # type unstable
```
We are getting a little ahead of ourselves in this lab, as understanding of these expressions is part of the future [lecture](@ref introspection) and [lab](@ref introspection_lab). Anyway the output basically shows what the compiler thinks of each variable in the code, albeit for us in less readable form than the original code. The more red the color is of the type info the less sure the inferred type is. Our main focus should be on the return type of the function which is just at the start of the code with the keyword `Body`. In the first case the return type is an `Int64`, whereas in the second example the compiler is unsure whether the type is `Float64` or `Int64`, marked as the `Union` type of the two. Fortunately for us this type instability can be fixed with a single line edit, but we will see later that it is not always the case.

!!! note "Type stability"
    Having a variable represented as `Union` of multiple types in a functions is a lesser evil than having `Any`, as we can at least enumerate statically the available options of functions to which to dynamically dispatch and in some cases there may be a low penalty.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Create a new function `polynomial_stable`, which is type stable and measure the difference in evaluation time. 

**HINTS**: 
- Ask for help on the `one` and `zero` keyword, which are often as a shorthand for these kind of functions.
- run the function with the argument once before running `@time` or use `@btime` if you have `BenchmarkTools` readily available in your environment
- To see some measurable difference with this simple function, a longer vector of coefficients may be needed.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@repl lab05_polynomial
function polynomial_stable(a, x)
    accumulator = zero(x)
    for i in length(a):-1:1
        accumulator += x^(i-1) * a[i]
    end
    accumulator
end
```

```@repl lab05_polynomial
@code_warntype polynomial_stable(a, x)  # type stable
@code_warntype polynomial_stable(af, xf) # type stable
```

```@repl lab05_polynomial
polynomial(af, xf) #hide
polynomial_stable(af, xf) #hide
@time polynomial(af, xf)
@time polynomial_stable(af, xf)
```

Only really visible when evaluating multiple times.
```@repl lab05_polynomial
using BenchmarkTools
@btime polynomial($af, $xf)
@btime polynomial_stable($af, $xf)
```
Difference only a few nanoseconds.


*Note*: Recalling homework from lab 1. Adding `zero` also extends this function to the case of `x` being a matrix, see `?` menu.
```@raw html
</p></details>
```

Code stability issues are something unique to Julia, as its JIT compilation allows it to produce code that contains boxed variables, whose type can be inferred during runtime. This is one of the reasons why interpreted languages are slow to run but fast to type. Julia's way of solving it is based around compiling functions for specific arguments, however in order for this to work without the interpreter, the compiler has to be able to infer the types.

There are other problems (such as repeated allocations, bad design patterns - arrays of struct, *others, may come from the performance gotchas*), that you can learn to spot in your code, however the code stability issues are by far the most common, for beginner users of Julia wanting to squeeze more out of it.

Sometimes `@code_warntype` shows that the function's return type is unstable without any hints to the possible problem, fortunately for such cases a more advanced tools such as [`Cthuhlu.jl`](https://github.com/JuliaDebug/Cthulhu.jl) or [`JET.jl`](https://github.com/aviatesk/JET.jl) have been developed and we will cover it in the next lecture. *we could use it in the ecosystem*

## Benchmarking
In the last exercise we have encountered the problem of timing of code to see if we have made any progress in speeding it up. Throughout the course we have advertised the use of the `BenchmarkTools` package, which provides easy way to test your code multiple times. In this lab we will focus on some advanced usage tips and gotchas that you may encounter while using it. *Furthermore in the homework you will create an code scalability benchmark.*


*BIG TODO HERE*


## Profiling
Profiling in Julia is part of the standard library in the `Profile` module. It implements a fairly simple sampling based profiler, which in a nutshell asks at regular intervals, where the code execution is currently at. As a result we get an array of stacktraces (= chain of function calls), which allow us to make sense of where the execution spent the most time. The number of samples, that can be stored and the period in seconds can be checked after loading `Profile` into the session with the `init()` function.

```@repl lab05_polynomial
using Profile
Profile.init()
```

The same function, but with keyword arguments, can be used to change these settings, however these settings are system dependent. For example on Windows, there is a known issue that does not allow to sample faster than at `0.003s` and even on Linux based system this may not do much. There are some further caveat specific to Julia:
- When running profile from REPL, it is usually dominated by the interactive part which spawns the task and waits for it's completion.
- Code has to be run before profiling in order to filter out all the type inference and interpretation stuff. (Unless compilation is what we want to profile.)
- When the execution time is short, the sampling may be insufficient -> run multiple times.

### Polynomial with scalars
Let's look at our favorite `polynomial` function or rather it's type stable variant `polynomial_stable` under the profiling lens.

```@repl lab05_polynomial
Profile.clear() # clear the last trace (does not have to be run on fresh start)
@profile polynomial_stable(af, xf)
Profile.print() # text based output of the profiler
```
Unless the machine that you run the code on is really slow, the resulting output contains nothing or only some internals of Julia's interactive REPL. This is due to the fact that our `polynomial` function take only few nanoseconds to run. When we want to run profiling on something, that takes only a few nanoseconds, we have to repeatedly execute the function.

```@repl lab05_polynomial
function run_polynomial_stable(a, x, n) 
    for _ in 1:n
        polynomial_stable(a, x)
    end
end

af = Float64.(rand(-10:10, 10)) # using longer polynomial

run_polynomial_stable(af, xf, 10) #hide
Profile.clear()
@profile run_polynomial_stable(af, xf, Int(1e5))
Profile.print()
```

In order to get more of a visual feel for profiling, there are packages that allow you to generate interactive plots or graphs. In this lab we will use [`ProfileSVG.jl`](https://github.com/timholy/ProfileSVG.jl), which does not require any fancy IDE or GUI libraries.

```@repl lab05_polynomial
using ProfileSVG
ProfileSVG.save("./scalar_prof.svg") # can work with already create traces
```
![profile](./scalar_prof.svg)


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Let's compare this with the type unstable situation.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

First let's define the function that allows us to run the `polynomial` multiple times.
```@repl lab05_polynomial
function run_polynomial(a, x, n) 
    for _ in 1:n
        polynomial(a, x)
    end
end
```

```@repl lab05_polynomial
run_polynomial(af, xf, 10) #hide
@profview run_polynomial(af, xf, Int(1e5)) # clears the profile for us
ProfileSVG.save("./scalar_prof_unstable.svg")
```
![profile_unstable](./scalar_prof_unstable.svg)


```@raw html
</p></details>
```

Other options for viewing profiler outputs
- [ProfileView](https://github.com/timholy/ProfileView.jl) - close cousin of `ProfileSVG`, spawns gtk window with interactive FlameGraph
- [VSCode](https://www.julia-vscode.org/docs/stable/release-notes/v0_17/#Profile-viewing-support-1) - always imported `@profview` macro, flamegraphs (js extension required), filtering, one click access to source code 
- [PProf](https://github.com/vchuravy/PProf.jl) - serializes the profiler output to protobuffer and loads it in `pprof` web app, graph visualization of stacktraces


## Applying fixes
We have noticed that no matter if the function is type stable or unstable the majority of the computation falls onto the power function `^` and there is a way to solve this using a clever technique called Horner schema[^1], which uses distributive and associative rules to convert the sum of powers into an incremental multiplication of partial results.


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Rewrite the `polynomial` function using the Horner schema/method[^1]. Moreover include the type stability fixes from `polynomial_stable` You should get some *#x* speed up when measured against the old implementation (measure `polynomial` against `polynomial_stable`.

**BONUS**: Profile the new method and compare the differences in traces.

[^1]: Explanation of the Horner schema can be found on [https://en.wikipedia.org/wiki/Horner%27s\_method](https://en.wikipedia.org/wiki/Horner%27s_method).

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@repl lab05_polynomial
function polynomial(a, x)
    accumulator = a[end] * one(x)
    for i in length(a)-1:-1:1
        accumulator = accumulator * x + a[i]
    end
    accumulator  
end
```

Speed up:
- 42ns -> 12ns ~ 3.5x on integer valued input 
- 420ns -> 12ns ~ 15x on real valued input

```@repl lab05_polynomial
@btime polynomial($(Int.(af)), $(Int(xf)))
@btime polynomial_stable($(Int.(af)), $(Int(xf)))
@btime polynomial($af, $xf)
@btime polynomial_stable($af, $xf)
```
These numbers will be different on different HW.

**BONUS**: The profile trace does not even contain the calling of mathematical operators and is mainly dominated by the iteration utilities. In this case we had to increase the number of runs to `1e6` to get some meaningful trace.

```@repl lab05_polynomial
run_polynomial(af, xf, 10) #hide
@profview run_polynomial(af, xf, Int(1e6))
ProfileSVG.save("./scalar_prof_horner.svg")
```
![profile_horner](./scalar_prof_horner.svg)
```@raw html
</p></details>
```

## Ecosystem debugging
Let's now apply what we have learned so far on the much bigger codebase of our `Ecosystem` and `EcosystemCore` packages.

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
![ecosystem_unstable](./ecosystem.svg)


By investigating the top bars we see that most of the time is spend inside `EcosystemCore.find_rand`, either when called from `EcosystemCore.find_food` or `EcosystemCore.find_mate`.

```julia
# original
function EcosystemCore.find_rand(f, w::World)
    xs = filter(f, w.agents |> values |> collect)
    isempty(xs) ? nothing : sample(xs)
end
```

Looking at the original code, we may not know exactly what is the problem, however the red color indicates that the code may be type unstable. Let's confirm the suspicion by evaluation.

```julia
w = ws[1]                                           # get an instance of a wolf
f = x -> EcosystemCore.eats(w, x)                   # define the filter function used in the `find_rand`
EcosystemCore.find_rand(f, world)                   
@code_warntype EcosystemCore.find_rand(f, world)    # check type stability
```

Indeed we see that the function is type unstable. As a resulting in the other two functions to be type unstable
```julia
@code_warntype EcosystemCore.find_food(w, world) # unstable as expected
@code_warntype EcosystemCore.find_mate(w, world) # unstable as expected
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```

Try to fix the type instability in `EcosystemCore.find_rand` by redefining it in the current session, i.e. write function
```julia
function EcosystemCore.find_rand(f, w::World)
    ...
end
```
that has the same functionality, while not having return type of `Any`.

**HINT**: With the current design of the whole package we cannot really get anything better than `Union{Agent, Nothing}`

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
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
![ecosystem_stablish](./ecosystem_1.svg)

```@raw html
</p></details>
```


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```

How big of a performance penalty did we have to pay? Benchmark the simulation against the original version and the improved version.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
using BenchmarkTools

gs = [Grass(id, regrowth_time) for id in 1:n_grass];
ss = [Sheep(id, 2*Δenergy_sheep, Δenergy_sheep, sheep_reproduce, sheep_foodprob) for id in n_grass+1:n_grass+n_sheep];
ws = [Wolf(id, 2*Δenergy_wolf, Δenergy_wolf, wolf_reproduce, wolf_foodprob) for id in n_grass+n_sheep+1:n_grass+n_sheep+n_wolves];
world = World(vcat(gs, ss, ws));

# does not work with the setup (works only with one setup variable)
# https://github.com/JuliaCI/BenchmarkTools.jl/issues/44
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

```@raw html
</p></details>
```

### Tracking allocations
Memory allocation is oftentimes the most CPU heavy part of the computation, thus working with memory correctly, i.e. avoiding unnecessary allocation is key for a well performing code. In order to get a sense of how much memory is allocated at individual places of the your codebase, we can instruct Julia to keep track of the allocations with a command line option `--track-allocation={all|user}` *figure out what these options do*
- all
- user

After exiting, Julia will create a copy of each source file, that has been touched during execution and assign to each line the number of allocations in bytes.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```

Transform the simulation code above into a script. Run this script with Julia with the `--track-allocation={all|user}` option, i.e.
```bash
julia -L ./your_script.jl --track-allocation={all|user}
```

Investigate the results of allocation tracking inside `EcosystemCore` source files. Where is the line with the most allocations?
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

I would expect that the same piece of code that has been type unstable also shows the allocations - the line inside `find_rand` that contains `filter, collect, keys, etc.`. *CHECK*


```@raw html
</p></details>
```
