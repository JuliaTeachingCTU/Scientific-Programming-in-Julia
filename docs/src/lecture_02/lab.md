# Lab 02: Predator-Prey Agents

To practice Julia's multiple dispatch you will implement your own, simplified,
agent-based simulation of a *predator-prey model*.  The model will contain
*wolves*, *sheep*, and - to feed your sheep - some *grass*.
Your final result could look something like the plot below.

![img](pred-prey.png)

As you can see, in this model, the wolves unfortunately died out :(.

To get started we need a type hierarchy. In order to be able to extend this model
in later lectures we will structure them like this

```julia
abstract type AbstractAgent end
abstract type AbstractAnimal <: AbstractAgent end
abstract type AbstractPlant <: AbstractAgent end
```

Our `Grass` will be growing over time and it will need a certain amount of time
steps to fully grow such that it can be eaten. This has to be reflected in the
fields of our grass struct:
```julia
mutable struct Grass <: AbstractPlant
    fully_grown::Bool
    regrowth_time::Int
    countdown::Int
end
Grass(t) = Grass(true,t,t)
```

Most of the logic of our agent simulation will be located in the function
`agent_step!(::AbstractAgent, ::World)`.
Grass cannot grow in a void, hence we need the `World` in the `agent_step!`.
In our case this world will be simply a container for all our agents.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Header</header>
<div class="admonition-body">
```

Define a `World` struct that will hold all your `AbstractAgents` in a `Vector`.
Try to avoid fields with abstract types. Julia's compiler will not be able to
infer the type for those (which leads to type instabilities and performance
losses).

TODO: linking to the lecture would be nice here.

After that, implement the `agent_step!` method for `Grass`. It should decrease
the `countdown` until the `Grass` is fully grown.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
struct World{V<:Vector{<:AbstractAgent}}
    agents::V
end

function agent_step!(a::Grass, w::World)
    if !a.fully_grown
        if a.countdown <= 0
            a.fully_grown = true
            a.countdown = a.regrowth_time
        else
            a.countdown -= 1
        end
    end
end
```

```@raw html
</p></details>
```

Now you should be able to create a world and grow some grass!
```julia
grass = Grass(false,2,2)
world = World([grass])

agent_step!(grass, world)
agent_step!(grass, world)
```
And so on and so forth. Probably first create `agent_step!` just for `Sheep`
then generalize to `AbstractAnimal`.
Not sure if this is the best example. The only parts that make use of
dispatch are `eats!` and `agent_step!`...

```julia
mutable struct Sheep{T<:Real} <: AbstractAnimal
    energy::T
    Δenergy::T
    reproduction_prob::T
    food_prob::T
end
Sheep() = Sheep(10.0, 5.0, 0.5)

mutable struct Wolf{T<:Real} <: AbstractAnimal
    energy::T
    Δenergy::T
    reproduction_prob::T
    food_prob::T
end
Wolf() = Wolf(10.0, 2.0, 0.01)

function agent_step!(a::Grass, w::World)
    if !a.fully_grown
        if a.countdown <= 0
            a.fully_grown = true
            a.countdown = a.regrowth_time
        else
            a.countdown -= 1
        end
    end
    return a
end

function agent_step!(a::A, w::World) where A<:AbstractAnimal
    a.energy -= 1
    dinner = find_food(a,w)
    eat!(a, dinner, w)
    if a.energy < 0
        kill_agent!(a,w)
        return
    end
    if rand() <= a.reproduction_prob
        reproduce!(a,w)
    end
    return a
end

function find_food(a::T, w::World) where T<:AbstractAnimal
    if rand() <= a.food_prob
        as = filter(x->isa(x,eats(T)), w.agents)
        isempty(as) ? nothing : sample(as)
    end
end

eats(::Type{<:Sheep}) = Grass
eats(::Type{<:Wolf}) = Sheep

function eat!(wolf::Wolf, sheep::Sheep, w::World)
    kill_agent!(sheep,w)
    wolf.energy += wolf.Δenergy
end
function eat!(sheep::Sheep, grass::Grass, w::World)
    if grass.fully_grown
        grass.fully_grown = false
        sheep.energy += sheep.Δenergy
    end
end
eat!(a,b,w) = nothing

function reproduce!(a::AbstractAnimal, w::World)
    a.energy /= 2
    push!(w.agents, deepcopy(a))
end

kill_agent!(a::AbstractAnimal, w::World) = deleteat!(w.agents, findall(x->x==a, w.agents))
```
