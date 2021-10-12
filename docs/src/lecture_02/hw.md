# Homework 2: Predator-Prey Agents

In this lab you will continue working on your agent simulation. If you did not
manage to finish the homework, do not worry, you can use [this
script](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/master/docs/src/lecture_02/Lab02Ecosystem.jl)
which contains all the functionality we developed in the lab.
```@setup hw02
projdir = dirname(Base.active_project())
include(joinpath(projdir,"src","lecture_02","Lab02Ecosystem.jl"))
```

## How to submit?

Put all your code (including your or the provided solution of lab 2) in a
script named `hw.jl` alongside with the `Project.toml` and `Manifest.toml` of
the environment. Please only include packages in the environment that are
necessary for the homework.  Create a `.zip` archive of the three files and
send it to the lab instructor, who has assigned the task, via email (contact
emails are located on the [homepage](@ref emails) of the course).


## Counting Agents

To monitor the different populations in our world we need a function that
counts each type of agent. For `AbstractAnimal`s we simply have to count how
many of each type are currently in our `World`. In the case of `AbstractPlant`s
we will use the fraction of `size(plant)/max_size(plant)` as a measurement
quantity.

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Compulsory Homework (2 points)</header>
<div class="admonition-body">
```
1. Implement a function `agent_count` that can be called on a single
   `AbstractAgent` and returns a number between $(0,1)$ (i.e. always `1` for animals;
   and `size(plant)/max_size(plant)` for plants).

2. Add a method for a vector of agents `Vector{<:AbstractAgent}` which sums all
   agent counts.

3. Add a method for a `World` which returns a dictionary
   that contains pairs of `Symbol`s and the agent count like below:

```@setup hw02
agent_count(p::Plant) = size(p)/max_size(p)
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

# Plot your simulation

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Voluntary Exercise (voluntary)</header>
<div class="admonition-body">
```
Using the world below, run few `world_step!`s.  Plot trajectories of the agents
count over time. Can you tweak the parameters such that you get similar oscillations
as in the plot from [lab 2](@ref lab02)?
```@example hw02
n_grass = 200
m_size  = 10

n_sheep  = 10
Δe_sheep = 0.2
e_sheep  = 4.0
pr_sheep = 0.8
pf_sheep = 0.6

n_wolves = 2
Δe_wolf  = 8.0
e_wolf   = 10.0
pr_wolf  = 0.1
pf_wolf  = 0.2

gs = [Grass(id,m_size,m_size) for id in 1:n_grass]
ss = [Sheep(id,e_sheep,Δe_sheep,pr_sheep,pf_sheep) for id in (n_grass+1):(n_grass+n_sheep)]
ws = [Wolf(id,e_wolf,Δe_wolf,pr_wolf,pf_wolf) for id in (n_grass+n_sheep+1):(n_grass+n_sheep+n_wolves)]
w  = World(vcat(gs,ss,ws))
nothing # hide
```
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@example hw02
counts = Dict(n=>[c] for (n,c) in agent_count(w))
for _ in 1:100
    world_step!(w)
    for (n,c) in agent_count(w)
        push!(counts[n],c)
    end
end

using Plots
plt = plot()
for (n,c) in counts
    plot!(plt, c, label="$n", lw=2)
end
plt
```

```@raw html
</p></details>
```
