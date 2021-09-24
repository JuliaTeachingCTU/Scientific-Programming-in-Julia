# [Lab 2: Predator-Prey Agents](@id lab02)


## The Predator-Prey Model

In the next two labs you will implement your own, simplified, agent-based
simulation of a *predator-prey model*.  The model will contain *wolves*,
*sheep*, and - to feed your sheep - some *grass*.  Running and plotting your
final result could look something like the plot below.

![img](pred-prey.png)
As you can see, in this model, the wolves unfortunately died out :(.
In this lab we will first define what our agent simulation does on a high level.
Then you will write the core methods to make everything work.

In an agent simulation we assume that we have a bunch of agents (in our case
grass, sheep, and wolves) that act in some environment (we will call it a
*world*). At every iteration of the simulation each agent will perform a *step* in which it
performs some of the actions that it can take. For example, a *grass* agent will grow
a little at every step. A *sheep* agent will try to find some grass and
reproduce.

To get started we need a type hierarchy. The first abstract type `Agent` that
acts as the root of our tree.  All animals and plants will be subtypes of `Agent`.
There are different kinds of animals and plants so it makes sense to create an
`Animal` type which will be the supertype of all animals. The same is
true for `Plant`s. Finally, we need a simple world which consist of
a dictionary of agent IDs to agents, and a field that holds the current maximum
ID (so that we can generate new ones).
```@example non_parametric_agents
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
```
Note that the `World` struct is not mutable because it only holds a pointer to
the `agents` vector. Hence, mutating the contents of the vector does not mutate
the world. However, changing the pointer to a completely different vector would
mutate the struct but this is not what we want in this case.

The function `simulate!` which will run our simulation for a number of
iterations `iters` will apply the function `agent_step!` to each agent.
Note that this function assumes that all `Agent`s have a _**unique**_ `id`.
```@example non_parametric_agents
function simulate!(world::World, iters::Int; callbacks=[])
    for i in 1:iters
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
        for cb in callbacks
            cb(world)
        end
    end
end
nothing # hide
```
The `agent_step!` function will have multiple methods and is the first function
in this lab that uses dispatch. The method for plants checks if a plant is
fully grown. If it is, nothing happens. If it is not fully grown, the plant
will grow a little each time `agent_step!` is called. When its growth countdown
has reached zero it becomes a fully grown plant.
```@example non_parametric_agents
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
nothing # hide
```
The `agent_step!` method for animals is different. At the beginning of each step
an animal looses energy. Afterwards it tries to find some food, which it will
subsequently eat. If the animal then has less than zero energy it dies and is
removed from the world. If it has positive energy it will try to reproduce.
```@example non_parametric_agents
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
nothing # hide
```

## The `Grass` Agent
The first concrete type we implement is the basis of life in our simulation and
source of all energy: `Grass`.
Our `Grass` will be growing over time and it will need a certain amount of time
steps to fully grow before it can be eaten.
We will realise this with a `countdown` field. At every step of the simulation
the `countdown` will decrease until it reaches zero at which point the grass
is fully grown.  This has to be reflected in the fields of our grass struct:
```@example non_parametric_agents
mutable struct Grass <: Plant
    id::Int
    fully_grown::Bool
    regrowth_time::Int
    countdown::Int
end
```
Note that `Grass` is a mutable type because e.g. the `countdown` field will
change during the simulation.
Let us assume that all plants have at least the fields `id`, `fully_grown`,
`regrowth_time`, and `countdown`, because all plants need some time to grow. If
this is the case we can make `Grass` a subtype of `Plant` (via `<:`) and
define a common interface for all `Plant`s.
```@example non_parametric_agents
# get field values
id(a::Agent) = a.id  # every agent has an ID so we can just define id for Agent here
fully_grown(a::Plant) = a.fully_grown
countdown(a::Plant) = a.countdown

# set field values
# (exclamation marks `!` indicate that the function is mutating its arguments)
fully_grown!(a::Plant, b::Bool) = a.fully_grown = b
countdown!(a::Plant, c::Int) = a.countdown = c
incr_countdown!(a::Plant, Δc::Int) = countdown!(a, countdown(a)+Δc)

# reset plant couter once it's grown
reset!(a::Plant) = a.countdown = a.regrowth_time
nothing # hide
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
1. Define a constructor for `Grass` which, given only an ID and a regrowth
   time $t$, will create an instance of `Grass` that is not fully grown and has
   a randomly initialized `countdown` in the range $(1,t)$.
2. Create some `Grass` agents inside a `World`
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
The constructor for grass with random growth countdown:
```@example non_parametric_agents
Grass(id,t) = Grass(id,false, t, rand(1:t))
nothing # hide
```
Creation of a world with a few grass agents:
```@repl non_parametric_agents
w = World([Grass(id,3) for id in 1:2])
simulate!(w,3,callbacks=[w->@show w])
```
```@raw html
</p></details>
```

## Sheep eat grass
Our simulated `Sheep` will have a certain amount of energy $E$, a reproduction
probability $p_r$, and a probablity to find food $p_f$ in each iteration of our
simulation. Additionally, each sheep with get a certain amout of energy $\Delta
E$ from eating a `Grass`. The corresponding struct then looks like this
```@example non_parametric_agents
mutable struct Sheep <: Animal
    id::Int
    energy::Float64
    Δenergy::Float64
    reproduction_prob::Float64
    food_prob::Float64
end
```
Again we will use `Sheep` as a generic example for an `Animal` which
leaves us with the interface below. We only have setters for `energy` because
all other fields of our animals will stay constant.
```@example non_parametric_agents
# get field values
energy(a::Animal) = a.energy
Δenergy(a::Animal) = a.Δenergy
reproduction_prob(a::Animal) = a.reproduction_prob
food_prob(a::Animal) = a.food_prob

# set field values
energy!(a::Animal, e) = a.energy = e
incr_energy!(a::Animal, Δe) = energy!(a, energy(a)+Δe)
nothing # hide
```

In every iteration of the simulation each sheep will get a chance to eat some
grass. The process of one animal eating a plant (or another animal) will be
implemented via the `eat!(a::Agent,b::Agent,::World)` function.
Calling the function will cause agent `a` to eat agent `b`, possibly mutating
them and the world. The `eat!` function will do something different for different
input types and is our first practical example of [multiple dispatch](@ref
multiple_dispatch).
The `eat!` function is part of our interface and we will have to implement a
special methods for each new type that we introduce.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement a function `eat!(::Sheep, ::Grass, ::World)` which increases the sheep's
energy by $\Delta E$ and sets `fully_grown` of the grass to `false`.
Note that you do not yet need the world in this function. It is needed later
for the case of wolves eating sheep.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example non_parametric_agents
function eat!(sheep::Sheep, grass::Grass, w::World)
    if fully_grown(grass)
        fully_grown!(grass, false)
        incr_energy!(sheep, Δenergy(sheep))
    end
end
nothing # hide
```
```@raw html
</p></details>
```
Below you can see how a fully grown grass is eaten by a sheep.  The sheep's
energy changes and the `fully_grown` field of the grass becomes `false`.
```@repl non_parametric_agents
grass = Grass(1,true,5,5)
sheep = Sheep(2,10.0,5.0,0.1,0.1)
world = World([grass, sheep])
eat!(sheep,grass,world);
world
```
Note that the order of the arguments has a meaning here. Calling
`eat!(grass,sheep,world)` results in a `MethodError` which is great, because
`Grass` cannot eat `Sheep`.
```@repl non_parametric_agents
eat!(grass,sheep,world);
```



## Wolves eat sheep
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Next, implement a `Wolf` with the same properties as the sheep ($E$, $\Delta
E$, $p_r$, and $p_f$) as well as the correspoding `eat!` method which increases
the wolf's energy and kills the sheep (i.e. removes the sheep from the world).

Hint: You can use `delete!` to remove agents from the dictionary in your world.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@example non_parametric_agents
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

kill_agent!(a::Animal, w::World) = delete!(w.agents, a.id)
nothing # hide
```
```@raw html
</p></details>
```
With a correct `eat!` method you should get results like this:
```@repl non_parametric_agents
grass = Grass(1,true,5,5);
sheep = Sheep(2,10.0,5.0,0.1,1.0);
wolf  = Wolf(3,20.0,10.0,0.1,1.0);
world = World([grass, sheep, wolf])
eat!(wolf,sheep,world);
world
```
The sheep is removed from the world and the wolf's energy increased by $\Delta E$.


## Finding food for sheep

```@setup non_parametric_agents
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
```

The next mechanism in our simulation models an animal's search for food.  For
example, a sheep can only try to eat if the world currently holds some grass.
The process of finding food for a given animal will be implemented by the
function `find_food(s::Sheep, ::World)`.
The food probability $p_f$ is the probability that a sheep finds food at a
given step of the simulation.  Hence, `find_food` will either return `nothing`
(if the sheep does not find grass) or sample a random `Grass` from all
available `Grass` agents.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```

Implement the method `find_food(::Sheep, ::World)` which first returns either a
`Grass` (sampled randomly from all `Grass`es) with the given food probability
$p_f$ or returns `nothing`.

1. Hint: For the functional programming way of coding this can use `filter` and
   `isa` to filter for a certain type and `StatsBase.sample` to choose a random
   element from a vector. You can get an `Iterator` of values of your dictionary
   via `values` which you might want to `collect` before passing it to `sample`.
2. Hint: You could also program this with a for-loop that iterates over your
   agents in a random order.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
using StatsBase  # needed for `sample`
# you can install it by typing `]add StatsBase` in the REPL

function find_food(a::Sheep, w::World)
    if rand() <= food_prob(a)
        as = filter(x->isa(x,Grass), w.agents |> values |> collect)
        isempty(as) ? nothing : sample(as)
    end
end
```
```@raw html
</p></details>
```
To test your function your can create sheep with different $p_f$.
A sheep with $p_f=1$ will always find some food if there is some in the world,
so you should get a result like below.
```@repl non_parametric_agents
grass = Grass(1,true,5,0);
sheep = Sheep(2,10.0,5.0,1.0,1.0);
wolf  = Wolf(3,20.0,10.0,1.0,1.0);
world = World([grass, sheep, wolf])

dinner = find_food(sheep,world)
eat!(sheep,dinner,world);
sheep
world
```


## Finding food for wolves
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement a function `find_food(::Wolf, ::World)` which returns either a
`Sheep` (with the given food probability $p_f$) or returns `nothing`.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
function find_food(a::Wolf, w::World)
    if rand() <= food_prob(a)
        as = filter(x->isa(x,Sheep), w.agents |> values |> collect)
        isempty(as) ? nothing : sample(as)
    end
end
```
```@raw html
</p></details>
```


## General food finding
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Identify the code duplications between `find_food(::Sheep,::World)` and
`find_food(::Wolf,::World)` and generalize the function to
`find_food(::Animal, ::World)`.

Hint: You can introduce a new function `eats(::Agent)::Agent`
which specifies which type of agent eats another type of agent.

Once you have done this, remove the two more specific functions for sheep and
wolves and restart the REPL once you have deleted those old methods to clean up
your namespace.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
function find_food(a::Animal, w::World)
    if rand() <= food_prob(a)
        as = filter(x->eats(a,x), w.agents |> values |> collect)
        isempty(as) ? nothing : sample(as)
    end
end

eats(::Sheep,::Grass) = true
eats(::Wolf,::Sheep) = true
eats(::Agent,::Agent) = false
```
```@raw html
</p></details>
```


## Eating nothing
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
What happens if you call `eat!(wolf, find_food(wolf,world), world)` and there
are no sheep anymore? Or if the wolf's $p_f<1$?

Consider the world below and the for-loop which represents a simplified
simulation in which the wolf tries to eat a sheep in each iteration.
Why does it fail?
```julia
sheep = [Sheep(id,10.0,5.0,1.0,1.0) for id in 1:2]
wolf  = Wolf(3,20.0,10.0,1.0,1.0)
world = World(vcat(sheep, [wolf]))

for _ in 1:4
    dinner = find_food(wolf,world)
    eat!(wolf,dinner,world)
    @show world
end
```
Hint: You can try to overload the `eat!` function appropriately.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@example non_parametric_agents
# make sure any animal can also eat `nothing`
eat!(a::Animal,b::Nothing,w::World) = nothing

sheep = [Sheep(id,10.0,5.0,1.0,1.0) for id in 1:2]
wolf  = Wolf(3,20.0,10.0,1.0,1.0)
world = World(vcat(sheep, [wolf]))

for _ in 1:4
    local dinner # hide
    @show world
    dinner = find_food(wolf,world)
    eat!(wolf,dinner,world)
end
```

```@raw html
</p></details>
```

## Reproduction
Currently our animals can only eat. In our simulation we also want them to
reproduce. We will do this by adding a `reproduce!` method to `Animal`.
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Write a function `reproduce!` that takes an `Animal` and a `World`.
Reproducing will cost an animal half of its energy and then add an almost
identical copy of the given animal to the world.  The only thing that is
different from parent to child is the ID. You can simply Increase the `max_id`
of the world by one and use that as the new ID for the child.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```julia
function reproduce!(a::Animal, w::World)
    energy!(a, energy(a)/2)
    new_id = w.max_id + 1
    â = deepcopy(a)
    â.id = new_id
    w.agents[id(â)] = â
    w.max_id = new_id
end
```
You can avoid mutating the `id` field by reconstructing the child from scratch:
```@example non_parametric_agents
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
```@raw html
</p></details>
```


## Finally! `simulate!`

With all our functions in place we can finally run the `simulate!` function.

Some grass that is growing:
```@example non_parametric_agents
gs = [Grass(id,3) for id in 1:3]
world = World(gs)
simulate!(world, 4, callbacks=[w->@show w])
```

Some sheep that are reproducing and then dying:
```@example non_parametric_agents
ss = [Sheep(id,5.0,2.0,1.0,1.0) for id in 1:2]
world = World(ss)
simulate!(world, 3, callbacks=[w->@show w])
```

All of it together!
```@example non_parametric_agents
gs = [Grass(id,5) for id in 1:5]
ss = [Sheep(id,10.0,5.0,0.5,0.5) for id in 6:10]
ws = [Wolf(id,20.0,10.,0.1,0.1) for id in 11:12]

world = World(vcat(gs, ss, ws))
simulate!(world, 10, callbacks=[w->@show w])
```


The code for this lab is inspired by the predator-prey model of
[`Agents.jl`](https://juliadynamics.github.io/Agents.jl/v3.5/examples/predator_prey/)
