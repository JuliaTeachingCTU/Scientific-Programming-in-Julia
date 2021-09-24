# Homework 2: Predator-Prey Agents

In this lab you will continue working on your agent simulation. If you did not
manage to finish the homework, do not worry, you can use the code below which
contains all the functionality we developed in the lab.
```@example hw02
using StatsBase

abstract type Agent end
abstract type Animal <: Agent end
abstract type Plant <: Agent end

mutable struct World{A<:Agent}
    agents::Dict{Int,A}
    max_id::Int
end
function World(agents::Vector{<:Agent})
    World(Dict(id(a)=>a for a in agents), maximum(id.(agents)))
end

# optional: you can overload the `show` method to get custom
# printing of your World
function Base.show(io::IO, w::World)
    println(io, typeof(w))
    for (_,a) in w.agents
        println(io,"  $a")
    end
end

function world_step!(world::World)
    # make sure that we only iterate over IDs that already exist in the 
    # current timestep this lets us safely add agents
    ids = deepcopy(keys(world.agents))

    for id in ids
        # agents can be killed by other agents, so make sure that we are
        # not stepping dead agents forward
        !haskey(world.agents,id) && continue

        a = world.agents[id]
        agent_step!(a,world)
    end
end

function agent_step!(a::Plant, w::World)
    if size(a) != max_size(a)
        grow!(a)
    end
end

function agent_step!(a::Animal, w::World)
    incr_energy!(a,-1)
    if rand() <= food_prob(a)
        dinner = find_food(a,w)
        eat!(a, dinner, w)
    end
    if energy(a) <= 0
        kill_agent!(a,w)
        return
    end
    if rand() <= reproduction_prob(a)
        reproduce!(a,w)
    end
    return a
end

mutable struct Grass <: Plant
    id::Int
    size::Int
    max_size::Int
end

mutable struct Sheep <: Animal
    id::Int
    energy::Float64
    Δenergy::Float64
    reproduction_prob::Float64
    food_prob::Float64
end

mutable struct Wolf <: Animal
    id::Int
    energy::Float64
    Δenergy::Float64
    reproduction_prob::Float64
    food_prob::Float64
end

id(a::Agent) = a.id  # every agent has an ID so we can just define id for Agent here

Base.size(a::Plant) = a.size
max_size(a::Plant) = a.max_size
grow!(a::Plant) = a.size += 1

# get field values
energy(a::Animal) = a.energy
Δenergy(a::Animal) = a.Δenergy
reproduction_prob(a::Animal) = a.reproduction_prob
food_prob(a::Animal) = a.food_prob

# set field values
energy!(a::Animal, e) = a.energy = e
incr_energy!(a::Animal, Δe) = energy!(a, energy(a)+Δe)

function eat!(a::Sheep, b::Grass, w::World)
    incr_energy!(a, size(b)*Δenergy(a))
    kill_agent!(b,w)
end
function eat!(wolf::Wolf, sheep::Sheep, w::World)
    incr_energy!(wolf, energy(sheep)*Δenergy(wolf))
    kill_agent!(sheep,w)
end
eat!(a::Animal,b::Nothing,w::World) = nothing

kill_agent!(a::Plant, w::World) = a.size = 0
kill_agent!(a::Animal, w::World) = delete!(w.agents, id(a))

function find_food(a::Animal, w::World)
    as = filter(x->eats(a,x), w.agents |> values |> collect)
    isempty(as) ? nothing : sample(as)
end

eats(::Sheep,::Grass) = true
eats(::Wolf,::Sheep) = true
eats(::Agent,::Agent) = false

function reproduce!(a::A, w::World) where A<:Animal
    energy!(a, energy(a)/2)
    a_vals = [getproperty(a,n) for n in fieldnames(A) if n!=:id]
    new_id = w.max_id + 1
    â = A(new_id, a_vals...)
    w.agents[id(â)] = â
    w.max_id = new_id
end
nothing # hide
```

## Counting Agents

To monitor the different populations in our world we need a function that
counts each type of agent. For `AbstractAnimal`s we simply have to count how
many of each type are currently in our `World`. In the case of `AbstractPlant`s
we will use the fraction of `size(plant)/max_size(plant)` as a measurement
quantity.

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (2 points)</header>
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
grass2 = Grass(2,1,5);
sheep = Sheep(3,10.0,5.0,1.0,1.0);
wolf  = Wolf(4,20.0,10.0,1.0,1.0);
world = World([grass1, grass2, sheep, wolf]);

agent_count(world)  # one grass is fully grown; the other only 20% => 1.2
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
<header class="admonition-header">Exercise (voluntary)</header>
<div class="admonition-body">
```
Using the world below, run few `world_step!`s.  Plot trajectories of the agents
count over time. Can you tweak the parameters such that you get similar oscillations
as in the plot from [lab 2](@id lab02)?
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
