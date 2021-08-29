# Homework 2: Predator-Prey Agents

```@setup load_ecosystem
using Scientific_Programming_in_Julia
using Scientific_Programming_in_Julia.Ecosystem: eat!, find_food, agent_count
```

In this homework we will continue working on our agent simulation.  All your
code must be in a single file called `Ecosystem.jl` containing all the type
definitions and functions we created in [Lab 2](@ref lab02) and the work you
do in this homework. Zip this file and upload it to BRUTE to receive your
points for the homework.

To monitor the different populations in our world we need a function that
counts each type of agent. For `AbstractAnimal`s we simply have to count how
many of each type are currently in our `World`. In the case of `AbstractPlant`s
we only want to count the ones that are fully grown.

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (2 points)</header>
<div class="admonition-body">
```
1. Implement a function `agent_count` that can be called on a single
   `AbstractAgent` and returns either `0` or `1` (i.e. always `1` for animals;
   `1` for a fully grown plant and `0` otherwise).

2. Add a method for a vector of agents `Vector{<:AbstractAgent}`.

3. Add a method for a `World` which returns a dictionary
   that contains pairs of `Symbol`s and the agent count like below:

```@repl load_ecosystem
grass1 = Grass(true,5.0,5.0);
grass2 = Grass(false,5.0,2.0);
sheep = Sheep(10.0,5.0,1.0,1.0);
wolf  = Wolf(20.0,10.0,1.0,1.0);
world = World([grass1, grass2, sheep, wolf]);

agent_count(world)  # the grass that is not fully grown is not counted
```

Hint: You can get the *name* of a type by using the `nameof` function:
```@repl load_ecosystem
nameof(Grass)
```
Use as much dispatch as you can! ;)

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
# first solution using foldl instead of a for loop
agent_count(g::AbstractPlant) = g.fully_grown ? 1 : 0
agent_count(::AbstractAnimal) = 1
agent_count(as::Vector{<:AbstractAgent}) = sum(agent_count,as)

function agent_count(w::World)
    function op(d::Dict,a::T) where T<:AbstractAgent
        n = nameof(T)
        if n in keys(d)
            d[n] += agent_count(a)
        else
            d[n] = agent_count(a)
        end
        return d
    end
    foldl(op, w.agents, init=Dict{Symbol,Int}())
end
```

```julia
# second solution with StastBase.countmap
countsym(g::T) where T<:AbstractPlant = g.fully_grown ? nameof(T) : :NoCount
countsym(::T) where T<:AbstractAnimal = nameof(T)

function agent_count(w::World)
    cs = StatsBase.countmap(countsym.(w.agents))
    delete!(cs,:NoCount)
end
```

```@raw html
</p></details>
```


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise (voluntary)</header>
<div class="admonition-body">
```
Using the world below, run a simple simulation with `7` iterations.  In each
iteration the wolf has to `find_food` and `eat!`.  Plot trajectories of the
agent count over time.

(Do not include this code in your submission to BRUTE)
```julia
grass = [Grass(true,5.0,5.0) for _ in 1:2];
sheep = [Sheep(10.0,5.0,1.0,1.0) for _ in 1:5];
wolf  = Wolf(20.0,10.0,1.0,1.0);
world = World(vcat([wolf], sheep, grass));
```
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@example load_ecosystem
grass = [Grass(true,5.0,5.0) for _ in 1:2];
sheep = [Sheep(10.0,5.0,1.0,1.0) for _ in 1:5];
wolf  = Wolf(20.0,10.0,1.0,1.0);
world = World(vcat([wolf], sheep, grass));

ns = nameof.(unique(typeof.(world.agents)))
counts = Dict(n=>[] for n in ns);
for _ in 1:7
    cs = agent_count(world)
    eat!(wolf, find_food(wolf,world), world)
    for (n,c) in cs
        push!(counts[n], c)
    end
end

using Plots
plt = plot();
for n in ns
    plot!(plt, counts[n], label="$n", lw=2, ylims=(0,5))
end
plt
```

```@raw html
</p></details>
```
