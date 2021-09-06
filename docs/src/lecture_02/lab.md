# [Lab 2: Predator-Prey Agents](@id lab02)

```@setup load_ecosystem
using Scientific_Programming_in_Julia
using Scientific_Programming_in_Julia.Ecosystem: eat!, find_food, count
```

In the next two labs you will implement your own, simplified, agent-based
simulation of a *predator-prey model*.  The model will contain *wolves*,
*sheep*, and - to feed your sheep - some *grass*.  Running and plotting your
final result could look something like the plot below.

![img](pred-prey.png)

As you can see, in this model, the wolves unfortunately died out :(.

## Creating the world
To get started we need a type hierarchy. In order to be able to extend this model
in later labs we will create an `AbstractAgent` that acts as the root of our tree.
All animals and plants will be subtypes of `AbstractAgent` (it is a good
convention to start the name of an abstract type with `Abstract...`).
There are different kinds of animals and plants so it makes sense to create an
`AbstractAnimal` type which will be the supertype of all animals. The same is
true for `AbstractPlant`s:
```julia
abstract type AbstractAgent end
abstract type AbstractAnimal <: AbstractAgent end
abstract type AbstractPlant <: AbstractAgent end
```

The first concrete type we implement is the basis of life in our simulation and
source of all energy: `Grass`.
Our `Grass` will be growing over time and it will need a certain amount of time
steps to fully grow before it can be eaten. This has to be reflected in the
fields of our grass struct:
```julia
mutable struct Grass <: AbstractPlant
    fully_grown::Bool
    regrowth_time::Int
    countdown::Int
end
# constructor for grass with random growth countdown
Grass(t) = Grass(false, t, rand(1:t))
```
Note that `Grass` is a subtype of `AbstractPlant`. Let us assume that all plants
have at least the fields `fully_grown`, `regrowth_time`, and `countdown`,
because all plants need some time to grow. If this is the case we can define
a common interface for all `AbstractPlant`s.
```julia
# get field values
fully_grown(a::AbstractPlant) = a.fully_grown
countdown(a::AbstractPlant) = a.countdown

# set field values
# (exclamation marks `!` indicate that the function is mutating its arguments)
fully_grown!(a::AbstractPlant, b::Bool) = a.fully_grown = b
countdown!(a::AbstractPlant, c::Int) = a.countdown = c
incr_countdown!(a::AbstractPlant, Δc::Int) = countdown!(a, countdown(a)+Δc)

# reset plant couter once its grown
reset!(a::AbstractPlant) = a.countdown = a.regrowth_time
```

Grass cannot grow in a void, hence we need a `World`.  In our simple case this
world will just be a container for all our agents.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Define a `World` struct that will hold all your `AbstractAgents` in a `Vector`
called `agents`.
Try to avoid fields with abstract types. Julia's compiler will not be able to
infer the type for those (which leads to type instabilities and performance
losses; see the [composite types section in the lecture](@ref composite_types)).
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```julia
struct World{T<:AbstractAgent}
    agents::Vector{T}
end

# if you want you can overload the `show` method to get custom printing of your World
function Base.show(io::IO, w::World)
    println(io, typeof(w))
    map(a->println(io,"  $a"),w.agents)
end
```
```@raw html
</p></details>
```

Now you should be able to create a world some grass in it that will soon be
eaten by sheep:
```@repl load_ecosystem
grass = Grass(5)
world = World([grass])
```

## Sheep eat grass
Our simulated `Sheep` will have a certain amount of energy $E$, a reproduction
probability $p_r$, and a probablity to find food $p_f$ in each iteration of our
simulation. Additionally, each sheep with get a certain amout of energy $\Delta
E$ from eating a `Grass`. The corresponding struct then looks like this
```julia
mutable struct Sheep{T<:Real} <: AbstractAnimal
    energy::T
    Δenergy::T
    reproduction_prob::T
    food_prob::T
end
```
Again we will use `Sheep` as a generic example for an `AbstractAnimal` which
leaves us with the interface below. We only have setters for `energy` because
all other fields of our animals will stay constant.
```julia
# get field values
energy(a::AbstractAnimal) = a.energy
Δenergy(a::AbstractAnimal) = a.Δenergy
reproduction_prob(a::AbstractAnimal) = a.reproduction_prob
food_prob(a::AbstractAnimal) = a.food_prob

# set field values
energy!(a::AbstractAnimal, e) = a.energy = e
incr_energy!(a::AbstractAnimal, Δe) = energy!(a, energy(a)+Δe)
```

In every iteration of the simulation each sheep will get a chance to eat some
grass. The process of one animal eating a plant (or another animal) will be
implemented via the `eat!(a::AbstractAgent,b::AbstractAgent,::World)` function.
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
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```julia
function eat!(sheep::Sheep, grass::Grass, w::World)
    if fully_grown(grass)
        fully_grown!(grass, false)
        incr_energy!(sheep, Δenergy(sheep))
    end
end
```
```@raw html
</p></details>
```
Below you can see how a fully grown grass is eaten by a sheep.  The sheep's
energy changes and the `fully_grown` field of the grass becomes `false`.
```@repl load_ecosystem
grass = Grass(true,5.0,5.0);
sheep = Sheep(10.0,5.0,0.1,0.1);
world = World([grass, sheep])
eat!(sheep,grass,world);
world
```
Note that the order of the arguments has a meaning here. Calling
`eat!(grass,sheep,world)` results in a `MethodError` which is great, because
`Grass` cannot eat `Sheep`.
```@repl load_ecosystem
grass = Grass(true,5.0,5.0);     # hide
sheep = Sheep(10.0,5.0,0.1,0.1); # hide
world = World([grass, sheep]);   # hide
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

Hint: You can use `findall` and `deleteat!` to remove agents from your world.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
mutable struct Wolf{T<:Real} <: AbstractAnimal
    energy::T
    Δenergy::T
    reproduction_prob::T
    food_prob::T
end

function eat!(wolf::Wolf, sheep::Sheep, w::World)
    kill_agent!(sheep,w)
    incr_energy!(wolf, Δenergy(wolf))
end

kill_agent!(a::AbstractAnimal, w::World) = deleteat!(w.agents, findall(x->x==a, w.agents))
```
```@raw html
</p></details>
```
With a correct `eat!` method you should get results like this:
```@repl load_ecosystem
grass = Grass(true,5.0,5.0);
sheep = Sheep(10.0,5.0,0.1,0.1);
wolf  = Wolf(20.0,10.0,0.1,0.1);
world = World([grass, sheep, wolf])
eat!(wolf,sheep,world);
world
```
The sheep is removed from the world and the wolf's energy increased by $\Delta E$.


## Finding food for sheep

The next mechanism in our simulation models an animal's search for food.  For
example, a sheep can only try to eat if the world currently holds some grass.
The process of finding food for a given animal will be implemented by the
function `find_food(a::AbstractAnimal, ::World)`. This function will either
return `nothing` or another animal that can be eaten by `a` with the given food
probability $p_f$.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement the method `find_food(::Sheep, ::World)` which returns either a
`Grass` (sampled randomly from all `Grass`es with the given food probability
$p_f$) or returns `nothing`.

Hint: You can use `StatsBase.sample` to choose a random element from a vector.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
function find_food(a::Sheep, w::World)
    if rand() <= food_prob(a)
        as = filter(x->isa(x,Grass), w.agents)
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
```@repl load_ecosystem
grass = Grass(true,5.0,5.0);
sheep = Sheep(10.0,5.0,1.0,1.0);
wolf  = Wolf(20.0,10.0,1.0,1.0);
world = World([grass, sheep, wolf]);

dinner = find_food(sheep,world)
eat!(sheep,dinner,world);
sheep
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
        as = filter(x->isa(x,Sheep), w.agents)
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
`find_food(::AbstractAnimal, ::World)`

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
function find_food(a::AbstractAnimal, w::World)
    if rand() <= food_prob(a)
        as = filter(x->eats(a,x), w.agents)
        isempty(as) ? nothing : sample(as)
    end
end

eats(::Sheep,::Grass) = true
eats(::Wolf,::Sheep) = true
eats(::AbstractAgent,::AbstractAgent) = false
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

Write a simple for-loop that runs `7` iterations of a simple simulation that
lets a wolf eat one sheep in each iteration with this given world:
```julia
sheep = [Sheep(10.0,5.0,1.0,1.0) for _ in 1:5]
wolf  = Wolf(20.0,10.0,1.0,0.0)
world = World(vcat(sheep, [wolf]))
```
Hint: You can try to overload the `eat!` function appropriately.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
# make sure any animal can also eat `nothing`
eat!(a::AbstractAnimal,b::Nothing,w::World) = nothing

for _ in 1:10
    dinner = find_food(wolf,world)
    eat!(wolf,dinner,world)
end
```

```@raw html
</p></details>
```
