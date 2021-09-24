# [Lab 3: Predator-Prey Agents](@id lab03)

```@setup non_parametric_agents
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

# optional code snippet: you can overload the `show` method to get custom
# printing of your World
function Base.show(io::IO, w::World)
    println(io, typeof(w))
    for (_,a) in w.agents
        println(io,"  $a")
    end
end

function simulate!(world::World, iters::Int; callbacks=[])
    for i in 1:iters
        for id in deepcopy(keys(world.agents))
            !haskey(world.agents,id) && continue
            a = world.agents[id]
            agent_step!(a,world)
        end
        for cb in callbacks
            cb(world)
        end
    end
end

function agent_step!(a::Plant, w::World)
    if !fully_grown(a)
        if countdown(a) <= 0
            fully_grown!(a,true)
            reset!(a)
        else
            incr_countdown!(a,-1)
        end
    end
    return a
end

function agent_step!(a::Animal, w::World)
    incr_energy!(a,-1)
    dinner = find_food(a,w)
    eat!(a, dinner, w)
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
    fully_grown::Bool
    regrowth_time::Int
    countdown::Int
end
Grass(id,t) = Grass(id,false, t, rand(1:t))

# get field values
id(a::Agent) = a.id
fully_grown(a::Plant) = a.fully_grown
countdown(a::Plant) = a.countdown

# set field values
# (exclamation marks `!` indicate that the function is mutating its arguments)
fully_grown!(a::Plant, b::Bool) = a.fully_grown = b
countdown!(a::Plant, c::Int) = a.countdown = c
incr_countdown!(a::Plant, Δc::Int) = countdown!(a, countdown(a)+Δc)

# reset plant couter once it's grown
reset!(a::Plant) = a.countdown = a.regrowth_time

mutable struct Sheep <: Animal
    id::Int
    energy::Float64
    Δenergy::Float64
    reproduction_prob::Float64
    food_prob::Float64
end

# get field values
energy(a::Animal) = a.energy
Δenergy(a::Animal) = a.Δenergy
reproduction_prob(a::Animal) = a.reproduction_prob
food_prob(a::Animal) = a.food_prob

# set field values
energy!(a::Animal, e) = a.energy = e
incr_energy!(a::Animal, Δe) = energy!(a, energy(a)+Δe)

function eat!(sheep::Sheep, grass::Grass, w::World)
    if fully_grown(grass)
        fully_grown!(grass, false)
        incr_energy!(sheep, Δenergy(sheep))
    end
end

mutable struct Wolf <: Animal
    id::Int
    energy::Float64
    Δenergy::Float64
    reproduction_prob::Float64
    food_prob::Float64
end

function eat!(wolf::Wolf, sheep::Sheep, w::World)
    kill_agent!(sheep,w)
    incr_energy!(wolf, Δenergy(wolf))
end
eat!(a::Animal,b::Nothing,w::World) = nothing

kill_agent!(a::Animal, w::World) = delete!(w.agents, id(a))

using StatsBase
function find_food(a::Animal, w::World)
    if rand() <= food_prob(a)
        as = filter(x->eats(a,x), w.agents |> values |> collect)
        isempty(as) ? nothing : sample(as)
    end
end

eats(::Sheep,::Grass) = true
eats(::Wolf,::Sheep) = true
eats(::Agent,::Agent) = false

function reproduce!(a::A, w::World) where A
    energy!(a, energy(a)/2)
    a_vals = [getproperty(a,n) for n in fieldnames(A) if n!=:id]
    new_id = w.max_id + 1
    â = A(new_id, a_vals...)
    w.agents[id(â)] = â
    w.max_id = new_id
end
```

In this lab we will look at two different ways of extending our agent
simulation to take into account that animals can have two different sexes:
*female* and *male*.

In the first part of the lab you will re-use the code from [lab 2](@ref lab02)
and create a new type of sheep (`⚥Sheep`) which has an additional field *sex*.
In the second part you will redesign the type hierarchy from scratch using
parametric types to make this agent system much more flexible and *julian*.

## Part I: Female & Male Sheep

The goal of the last part of the lab is to demonstrate the [forwarding
method](@ref forwarding_method) by implementing a sheep that can have two
different sexes and can only reproduce with another sheep of opposite sex.

This new type of sheep needs an additonal field `sex::Symbol` which can be either
`:male` or `:female`.
In OOP we would now simply inherit from `Sheep` and create a `⚥Sheep`
with an additional field. In Julia there is no inheritance - only subtyping of
abstract types.
As you cannot inherit from a concrete type in Julia, we will have to create a
wrapper type and forward all necessary methods. This is typically a sign of
unfortunate type tree design and should be avoided, but if you want to extend a
code base by an unforeseen type this forwarding of methods is a nice
work-around.  Our `⚥Sheep` type will simply contain a classic `sheep` and a
`sex` field
```@example non_parametric_agents
struct ⚥Sheep <: Animal
    sheep::Sheep
    sex::Symbol
end
⚥Sheep(id,E,ΔE,pr,pf,sex) = ⚥Sheep(Sheep(id,E,ΔE,pr,pf),sex)
nothing # hide
```

```@repl non_parametric_agents
⚥Sheep(1,1.0,1.0,1.0,1.0,:female)
```

In our case, the methods that have to be forwarded are `agent_step!`,
`reproduce!`, `eats` and `eat!`.  The custom reproduction behaviour will of
course be taken care of by a `reproduce!` function that does not just
forward but also contains specialized behaviour for the `⚥Sheep`.


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Forward the accessors `energy`, `energy!`, `reproduction_prob`, and `food_prob`,
as well as our core methods `eats` and `eat!` to `Sheep`.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example non_parametric_agents
id(g::⚥Sheep) = id(g.sheep)
energy(g::⚥Sheep) = energy(g.sheep)
energy!(g::⚥Sheep, ΔE) = energy!(g.sheep, ΔE)
reproduction_prob(g::⚥Sheep) = reproduction_prob(g.sheep)
food_prob(g::⚥Sheep) = food_prob(g.sheep)

eats(::⚥Sheep, ::Grass) = true
# eats(::⚥Sheep, ::PoisonedGrass) = true
eat!(s::⚥Sheep, g::Plant, w::World) = eat!(s.sheep, g, w)
nothing # hide
```
```@raw html
</p></details>
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement the `reproduce!` method for the `⚥Sheep`.  Note that you first
have to find another sheep of opposite sex in your `World`, and only if you
can find one you can reproduce.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example non_parametric_agents
mates(a::Plant, ::⚥Sheep) = false
mates(a::Animal, ::⚥Sheep) = false
mates(g1::⚥Sheep, g2::⚥Sheep) = g1.sex != g2.sex
function find_mate(g::⚥Sheep, w::World)
    ms = filter(a->mates(a,g), w.agents |> values |> collect)
    isempty(ms) ? nothing : sample(ms)
end

function reproduce!(s::⚥Sheep, w::World)
    m = find_mate(s,w)
    if !isnothing(m)
        energy!(s, energy(s)/2)
        vals = [getproperty(s.sheep,n) for n in fieldnames(Sheep) if n!=:id]
        new_id = w.max_id + 1
        ŝ = ⚥Sheep(new_id, vals..., rand(Bool) ? :female : :male)
        w.agents[id(ŝ)] = ŝ
        w.max_id = new_id
    end
end
```
```@raw html
</p></details>
```

```@example non_parametric_agents
f = ⚥Sheep(1,3.0,1.0,1.0,1.0,:female)
m = ⚥Sheep(2,4.0,1.0,1.0,1.0,:male)
w = World([f,m])
simulate!(w, 3, callbacks=[w->@show w])
```


## Part II: A new, parametric type hierarchy
