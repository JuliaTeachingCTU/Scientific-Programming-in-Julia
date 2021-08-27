# Lab 3: Predator-Prey Agents

```@setup load_ecosystem
using Scientific_Programming_in_Julia
using Scientific_Programming_in_Julia.Ecosystem: eat!, find_food, count
```

In this lab we will finalize our predator-prey agent simulation such that we
can simulate a number of steps in order to get a plot like below.

![img](pred-prey.png)


## Reproduction
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
The only other thing that our animals are able to do apart from eating will
be reproducing. Write a function `reproduce!` that take an `AbstractAnimal`
and a `World`. Reproducing will cost an animal half of its energy and add an
identical copy of the given animal to the world.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```julia
function reproduce!(a::AbstractAnimal, w::World)
    a.energy /= 2
    push!(w.agents, deepcopy(a))
end
```
```@raw html
</p></details>
```


## One step at a time
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
One iteration of our simulation will be carried out by a function called
`agent_step!(::AbstractAgent, ::World)`. Implement one method for `AbstractPlant`s
and one for `AbstractAnimal`s.

An `AbstractPlant` will grow if it is not fully grown (i.e. decrease growth
counter).  If the growth counter has reached zero, the `fully_grown` flag has
to be set to zero and the counter reset to `regrowth_time`.

An `AbstractAnimal` will loose one unit of energy in every step.  Then it will
try to find food and eat. After eating, if its energy is less than zero, the
animal dies.  If it is still alive, it will try to reproduce with the
probablitiy $p_r$.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```julia
function agent_step!(a::AbstractPlant, w::World)
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
```
```@raw html
</p></details>
```

## Simulate the world!
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
The last function we need to run our simulation just needs to run a number of
steps. In practice we often want varying logging behaviour which we can
implement nicely using *callbacks*. Implement a function
`simulate!(w::World, iters::Int; callbacks=[])`
which applies a number of callbacks of the form `callback(::World)`
after each iteration.

An exemplary callback could just log the agent count at ever step:
```julia
log_count(w::World) = @info agent_count(w)
```
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

cbs = [w->(@info agent_count(w))]
simulate!(w, 10, callbacks=cbs)
```



## Smarter callbacks through closures
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Often we want our callbacks to be executed only every $N$th step.  Implement a
function `every_nth(f::Function,n::Int)` that takes a function and uses a
closure to construct another function that only calls `f` every `n` calls to
the function `fn` that is returned by `every_nth(f,n)`.

Use `every_nth` to log the agent count every 5th step of your simulation and to
save every second agent count.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```julia
function every_nth(f::Function, n::Int)
    i = 1
    function callback(w::World)
        # display(i) # comment this out to see how the counter increases
        if i == n
            f(w)
            i = 1
        else
            i += 1
        end
    end
end

# construct a global variable to store the trajectories
counts = Dict(n=>[c] for (n,c) in agent_count(w))
# callback to save current counts
function _save(w::World)
    for (n,c) in agent_count(w)
        push!(counts[n],c)
    end
end

# create callbacks
logcb = every_nth(w->(@info agent_count(w)), 5)
savecb = every_nth(_save, 2)

simulate!(w, 200, callbacks=[logcb, savecb])

# you can now plot the trajectories like this
using Plots
plt = plot()
for (n,c) in counts
    plot!(plt, c, label="$n", lw=2)
end
display(plt)
```
```@raw html
</p></details>
```


## Poisoned Grass
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
In the previous exercises you have seen that multiple dispatch makes it easy to
add new methods to a given type (much easier than in OOP!).  In this exercise
you will see that it is just as easy to add a completely new type to our
hierarchy and reuse the methods that we have already defined (similar to
inheritance in OOP).

We have a few essential functions: `agent_step!`, `reproduce!`, `find_food`, `eats`,
and `eat!`. If you look at their type signatures you can see that the first
three already operate on any `AbstractAnimal`/`AbstractPlant`. This means that
for any subtype that has the expected fields (`energy`, `Δenergy`, `food_prob`,
and `reproduction_prob`) these functions already work.

The only methods we have to implement for a new animal or plant are the `eats`
and `eat!` methods.  So, lets implement a `PoisonedGrass` which will *decrease*
the energy of a sheep that ate it.

How much poisoned grass can you add to the simulation without wiping out the
sheep population?
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```julia
mutable struct PoisonedGrass <: AbstractPlant
    fully_grown::Bool
    regrowth_time::Int
    countdown::Int
end
PoisonedGrass(t) = PoisonedGrass(false, t, rand(1:t))

function eat!(sheep::Sheep, grass::PoisonedGrass, w::World)
     if grass.fully_grown
        grass.fully_grown = false
        sheep.energy -= sheep.Δenergy
    end
end

eats(::Sheep,::PoisonedGrass) = true
```
```@raw html
</p></details>
```
