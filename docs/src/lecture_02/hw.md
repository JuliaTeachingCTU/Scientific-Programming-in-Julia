# Homework 2: Predator-Prey Agents

In this lab you will continue working on your agent simulation. If you did not
manage to finish the homework, do not worry, you can use [this
script](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/2023W/docs/src/lecture_02/Lab02Ecosystem.jl)
which contains all the functionality we developed in the lab.
```@setup hw02
projdir = dirname(Base.active_project())
include(joinpath(projdir,"src","lecture_02","Lab02Ecosystem.jl"))
```

## How to submit?

Put all your code (including your or the provided solution of lab 2) in a script
named `hw.jl`.  Zip only this file (not its parent folder) and upload it to
BRUTE. Your file cannot contain any package dependencies.  For example, having a
`using Plots` in your code will cause the automatic evaluation to fail.



## Counting Agents

To monitor the different populations in our world we need a function that
counts each type of agent. For `Animal`s we simply have to count how
many of each type are currently in our `World`. In the case of `Plant`s
we will use the fraction of `size(plant)/max_size(plant)` as a measurement
quantity.

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Compulsory Homework (2 points)</header>
<div class="admonition-body">
```
1. Implement a function `agent_count` that can be called on a single
   `Agent` and returns a number between $(0,1)$ (i.e. always `1` for animals;
   and `size(plant)/max_size(plant)` for plants).

2. Add a method for a vector of agents `Vector{<:Agent}` which sums all
   agent counts.

3. Add a method for a `World` which returns a dictionary
   that contains pairs of `Symbol`s and the agent count like below:

```@setup hw02
agent_count(p::Plant) = p.size / p.max_size
agent_count(::Animal) = 1
agent_count(as::Vector{<:Agent}) = sum(agent_count,as)

function agent_count(w::World)
    function op(d::Dict,a::A) where A<:Agent
        n = nameof(A)
        if n in keys(d)
            d[n] += agent_count(a)
        else
            d[n] = agent_count(a)
        end
        return d
    end
    foldl(op, w.agents |> values |> collect, init=Dict{Symbol,Real}())
end
```

```@repl hw02
grass1 = Grass(1,5,5);
agent_count(grass1)

grass2 = Grass(2,1,5);
agent_count([grass1,grass2]) # one grass is fully grown; the other only 20% => 1.2

sheep = Sheep(3,10.0,5.0,1.0,1.0);
wolf  = Wolf(4,20.0,10.0,1.0,1.0);
world = World([grass1, grass2, sheep, wolf]);
agent_count(world)
```

Hint: You can get the *name* of a type by using the `nameof` function:
```@repl hw02
nameof(Grass)
```
Use as much dispatch as you can! ;)
```@raw html
</div></div>
```
