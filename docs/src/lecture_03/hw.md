# Homework 3

```@setup load_ecosystem
using Scientific_Programming_in_Julia
using Scientific_Programming_in_Julia.Ecosystem: eat!, find_food, count

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
```



In this homework we will extend our agent simulation with more powerful callbacks
and add a second type of grass. You can use the following snippet from the labs
as a base simulation script
```julia
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
Often we want our callbacks to be executed only every $N$th step. This can be
used to get less verbose logging or e.g. to write out checkpoints of your
simulation.
```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (1 point)</header>
<div class="admonition-body">
```
Implement a function `every_nth(f::Function,n::Int)` that takes a function and
uses a closure to construct another function that only calls `f` every `n`
calls to the function `fn` that is returned by `every_nth(f,n)`.

You can use `every_nth` to log and save the agent count only every couple of
steps of your simulation. Using `every_nth` will look like this:
```@repl load_ecosystem
# `@info agent_count(w)` is executed only every 5th call to logcb(w)
logcb = every_nth(w->(@info agent_count(w)), 5);

for i in 1:10
    logcb(w)
end
```
```@raw html
</div></div>
<details class = "solution-body" hidden>
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
```
```@raw html
</p></details>
```


## Poisoned grass

In the previous exercises you have seen that multiple dispatch makes it easy to
add new methods to a given type (much easier than in OOP!).  In this exercise
you will see that it is just as easy to add a completely new type to our
hierarchy and reuse the methods that we have already defined (similar to
inheritance in OOP).

We have a few essential functions: `agent_step!`, `reproduce!`, `find_food`, `eats`,
and `eat!`. If you look at their type signatures you can see that the first
three already operate on any `AbstractAnimal`/`AbstractPlant`. The same is true
for the accessors like `energy`/`energ!`. This means that
for any subtype that has the expected fields (i.e. `energy`,
`reproduction_prob`, etc.) these functions already work.

Therefore, the only methods we have to implement for a new animal or plant are the `eats`
and `eat!` methods.  So, lets implement a `PoisonedGrass` which will *decrease*
the energy of a sheep that ate it.

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (2 points)</header>
<div class="admonition-body">
```
Define a new subtype of `AbstractPlant` called `PoisonedGrass` which will
decrease a sheep's energy by $\Delta E$ if it is eaten.

Implement the functions `eat!` and `eats` in order to make the simulation work
with `PoisonedGrass`.

How much poisoned grass can you add to the simulation without wiping out the
sheep population?
```@raw html
</div></div>
<details class = "solution-body" hidden>
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


```@setup load_ecosystem
using Scientific_Programming_in_Julia
using Scientific_Programming_in_Julia.Ecosystem: eat!, find_food, count
```
