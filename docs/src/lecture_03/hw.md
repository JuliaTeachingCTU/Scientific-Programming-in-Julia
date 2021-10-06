# Homework 3

```@setup hw03
using EcosystemCore
using Scientific_Programming_in_Julia
```

In this homework we will add another species of plant to our simulation
(*poisoned mushrooms*) and practice the use of closures with callback
functions.

## Poisoned Mushrooms

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (1 point)</header>
<div class="admonition-body">
```
Implement a new species `Mushroom` which tricks a sheep into eating it by its
delicious looks but decreases the energy of the sheep by
`size(::Mushroom)*Δenergy(::Sheep)`.
```@raw html
</div></div>
```
Your new species should give you results like below
```@repl hw03
s = Sheep(1,2,2,1,1);
m = Mushroom(2,5);
w = World([s,m])
eat!(s,m,w);
w
```

## Callbacks & Closures

In many scientific frameworks we have to work with functions like `simulate!`
```@example hw03
function simulate!(w::World, iters::Int, callbacks=[])
    for _ in 1:iters
        world_step!(w)
        for cb in callbacks
            cb(w)
        end
    end
end
nothing # hide
```
which allow custom functionality within a larger simulation function.
For example, we might want to print out what the world looks like after
every time step. This could be done by passing a lambda function `w->(@show w)`
to `simulate!`.
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

```@raw html
</div></div>
```
You can use `every_nth` to log and save the agent count only every couple of
steps of your simulation. Using `every_nth` will look like this:
```@repl hw03
# `@info agent_count(w)` is executed only every 5th call to logcb(w)
logcb = every_nth(w->(@info agent_count(w)), 5);

for i in 1:10
    logcb(w)
end
```
