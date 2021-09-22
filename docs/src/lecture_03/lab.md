# [Lab 3: Predator-Prey Agents](@id lab03)

```@setup load_ecosystem
using Scientific_Programming_in_Julia
using Scientific_Programming_in_Julia.Ecosystem: eat!, find_food, count
```

In this lab we will finalize our predator-prey agent simulation such that we
can simulate a number of steps and get a plot like below.

![img](pred-prey.png)


## Reproduction
Currently our animals can only eat. In our simulation we also want them to
reproduce. We will do this by adding a `reproduce!` method to `AbstractAnimal`.
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Write a function `reproduce!` that takes an `AbstractAnimal` and a `World`.
Reproducing will cost an animal half of its energy and then add an identical copy of
the given animal to the world.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```julia
function reproduce!(a::AbstractAnimal, w::World)
    energy!(a, energy(a)/2)
    push!(w.agents, deepcopy(a))
end
```
```@raw html
</p></details>
```


## One step at a time
One iteration of our simulation will be carried out by a function called
`agent_step!(::AbstractAgent, ::World)`. Implement one method for `AbstractPlant`s
and one for `AbstractAnimal`s.

An `AbstractPlant` will grow if it is not fully grown (i.e. decrease growth
counter).  If the growth counter has reached zero, the `fully_grown` flag has
to be set to `true` and the counter reset to `regrowth_time`.

An `AbstractAnimal` will loose one unit of energy in every step.  Then it will
try to find food and eat. After eating, if its energy is less than zero, the
animal dies.  If it is still alive, it will try to reproduce with the
probablitiy $p_r$.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement the function `agent_step!` with specialized methods for `AbstractAnimal`s
and `AbstractPlant`s.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```julia
function agent_step!(a::AbstractPlant, w::World)
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

function agent_step!(a::AbstractAnimal, w::World)
    incr_energy!(a,-1)
    dinner = find_food(a,w)
    eat!(a, dinner, w)
    if energy(a) < 0
        kill_agent!(a,w)
        return
    end
    if rand() <= reproduction_prob(a)
        reproduce!(a,w)
    end
    return a
end
```
```@raw html
</p></details>
```

## Simulate the world!

The last function we need for our simulation just needs to run a number of
steps. In practice we often want varying logging behaviour which we can
implement nicely using *callbacks*.

An exemplary callback could just log the agent count at every step:
```julia
log_count(w::World) = @info agent_count(w)
```


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement a function `simulate!(w::World, iters::Int; callbacks=[])` which
runs a number of iterations (i.e. `agent_step!`s) and
applies passed callbacks of the form `callback(::World)` after each
iteration.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```julia
function simulate!(w::World, iters::Int; callbacks=[])
    for i in 1:iters
        for a in w.agents
            agent_step!(a,w)
        end
        for cb in callbacks
            cb(w)
        end
    end
end
```
```@raw html
</p></details>
```
Now lets try to run our first fully fledged simulation!  Below you can find
some parameters that will often result in nice oscillations like in the plot at
the beginning of the lab.
```@example load_ecosystem
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

gs = [Grass(true,regrowth_time,regrowth_time) for _ in 1:n_grass]
ss = [Sheep(2*Δenergy_sheep,Δenergy_sheep,sheep_reproduce, sheep_foodprob) for _ in 1:n_sheep]
ws = [Wolf(2*Δenergy_wolf,Δenergy_wolf,wolf_reproduce, wolf_foodprob) for _ in 1:n_wolves]

w = World(vcat(gs,ss,ws))

# construct a global variable to store agent counts at every step
counts = Dict(n=>[c] for (n,c) in agent_count(w))
# callback to save current counts
function save_agent_count(w::World)
    for (n,c) in agent_count(w)
        push!(counts[n],c)
    end
end

cbs = [
    w->(@info agent_count(w)),
    save_agent_count
]

simulate!(w, 200, callbacks=cbs)

# plot the count trajectories for every type of agent
using Plots
plt = plot()
for (n,c) in counts
    plot!(plt, c, label="$n", lw=2)
end
plt
```


## Female & Male Sheep

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
```julia
struct ⚥Sheep{T<:Real} <: AbstractAnimal
    sheep::Sheep{T}
    sex::Symbol
end
⚥Sheep(E,ΔE,pr,pf,sex) = ⚥Sheep(Sheep(E,ΔE,pr,pf),sex)
```

```@repl load_ecosystem
⚥Sheep(1.0,1.0,1.0,1.0,:female)
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
```julia
energy(g::⚥Sheep) = energy(g.sheep)
energy!(g::⚥Sheep, ΔE) = energy!(g.sheep, ΔE)
reproduction_prob(g::⚥Sheep) = reproduction_prob(g.sheep)
food_prob(g::⚥Sheep) = food_prob(g.sheep)

eats(::⚥Sheep, ::Grass) = true
eats(::⚥Sheep, ::PoisonedGrass) = true
eat!(s::⚥Sheep, g::AbstractPlant, w::World) = eat!(s.sheep, g, w)
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
```julia
mates(a::AbstractPlant, ::⚥Sheep) = false
mates(a::AbstractAnimal, ::⚥Sheep) = false
mates(g1::⚥Sheep, g2::⚥Sheep) = g1.sex != g2.sex
function find_mate(g::⚥Sheep, w::World)
    ms = filter(a->mates(a,g), w.agents)
    isempty(ms) ? nothing : sample(ms)
end

function reproduce!(s::⚥Sheep, w::World)
    m = find_mate(s,w)
    if !isnothing(m)
        energy!(s, energy(s)/2)
        # TODO: should probably mix s/m
        push!(w.agents, deepcopy(s))
    end
end
```
```@raw html
</p></details>
```
