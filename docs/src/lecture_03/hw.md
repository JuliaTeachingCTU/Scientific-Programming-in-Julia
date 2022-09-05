# Homework 3

use create world closure as homework

```@setup hw03
using EcosystemCore
using Scientific_Programming_in_Julia
```

In this homework we will add another species of plant to our simulation
(*poisoned mushrooms*) and practice the use of closures with callback
functions. The solution of lab 3 can be found
[here](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/master/docs/src/lecture_03/Lab03Ecosystem.jl). You can use this file and add the code that you write
for the homework to it.

## How to submit?

Put all your code (including your or the provided solution of lab 2)
in a script named `hw.jl`.  Zip only this file (not its parent folder) and
upload it to BRUTE.  Your file can contain one dependency `using StatsBase`,
but no other packages are can to be used.  For example, having a `using Plots`
in your code will cause the automatic evaluation to fail.


## Callbacks & Closures

In many scientific frameworks we have to work with functions like `simulate!`
(The `solve` function in
[`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) is a good example)
```@example hw03
function simulate!(w::World, iters::Int, cb=()->())
    for _ in 1:iters
        # In our case this loop is trivial. In more involved simulations this
        # will be more complicated ;)
        world_step!(w)
        cb()
    end
end
nothing # hide
```
which allow custom functionality within a larger simulation function.
For example, we might want to print out what the world looks like after
every time step. This could be done by passing a lambda function `w->(@show w)`
to `simulate!`.
Often we want our callbacks to be executed only every `n` steps. This can be
used to get less verbose logging or e.g. to write out checkpoints of your
simulation.
```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (1 point)</header>
<div class="admonition-body">
```
Implement a function `every_nth(f::Function,n::Int)` that takes an inner
function `f` and uses a closure to construct an outer function `g` that only
calls `f` every `n`th call to `g`. E.g. if `n=3` the inner function `f` be called
at the 3rd, 6th, 9th ... call to `g` (not at the 1st, 4th, 7th... call).

**Hint**: You can use splatting via `...` to pass on an unknown number of
arguments from the outer to the inner function.
```@raw html
</div></div>
```
You can use `every_nth` to log (or save) the agent count only every couple of
steps of your simulation. Using `every_nth` will look like this:
```@repl hw03
# `@info agent_count(w)` is executed only every 5th call to logcb(w)
logcb = every_nth(w->(@info agent_count(w)), 3);

logcb(w);  # x->(@info agent_count(w)) is not called
logcb(w);  # x->(@info agent_count(w)) is not called
logcb(w);  # x->(@info agent_count(w)) *is* called
```
