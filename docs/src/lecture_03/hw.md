# Homework 3

In this homework we will implement a function `find_food` and practice the use of closures.
The solution of lab 3 can be found
[here](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/2023W/docs/src/lecture_03/Lab03Ecosystem.jl).
You can use this file and add the code that you write for the homework to it.

## How to submit?

Put all your code (including your or the provided solution of lab 2)
in a script named `hw.jl`.  Zip only this file (not its parent folder) and
upload it to BRUTE.

```@setup block
projdir = dirname(Base.active_project())
include(joinpath(projdir,"src","lecture_03","Lab03Ecosystem.jl"))

function find_food(a::Animal, w::World)
    as = filter(x -> eats(a,x), w.agents |> values |> collect)
    isempty(as) ? nothing : rand(as)
end

eats(::Animal{Sheep},g::Plant{Grass}) = g.size > 0
eats(::Animal{Wolf},::Animal{Sheep}) = true
eats(::Agent,::Agent) = false

function every_nth(f::Function, n::Int)
    i = 1
    function callback(args...)
        # display(i) # comment this out to see out the counter increases
        if i == n
            f(args...)
            i = 1
        else
            i += 1
        end
    end
end

nothing # hide
```


## Agents looking for food

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework:</header>
<div class="admonition-body">
```
Implement a method `find_food(a::Animal, w::World)` returns one randomly chosen
agent from all `w.agents` that can be eaten by `a` or `nothing` if no food could
be found. This means that if e.g. the animal is a `Wolf` you have to return one
random `Sheep`, etc.

*Hint*: You can write a general `find_food` method for all animals and move the
parts that are specific to the concrete animal types to a separate function.
E.g. you could define a function `eats(::Animal{Wolf}, ::Animal{Sheep}) = true`, etc.

You can check your solution with the public test:
```@repl block
sheep = Sheep(1,pf=1.0)
world = World([Grass(2), sheep])
find_food(sheep, world) isa Plant{Grass}
```
```@raw html
</div></div>
```

## Callbacks & Closures

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework:</header>
<div class="admonition-body">
```
Implement a function `every_nth(f::Function,n::Int)` that takes an inner
function `f` and uses a closure to construct an outer function `g` that only
calls `f` every `n`th call to `g`. For example, if `n=3` the inner function `f` be called
at the 3rd, 6th, 9th ... call to `g` (not at the 1st, 2nd, 4th, 5th, 7th... call).

**Hint**: You can use splatting via `...` to pass on an unknown number of
arguments from the outer to the inner function.
```@raw html
</div></div>
```
You can use `every_nth` to log (or save) the agent count only every couple of
steps of your simulation. Using `every_nth` will look like this:
```@repl block
w = World([Sheep(1), Grass(2), Wolf(3)])
# `@info agent_count(w)` is executed only every 3rd call to logcb(w)
logcb = every_nth(w->(@info agent_count(w)), 3);

logcb(w);  # x->(@info agent_count(w)) is not called
logcb(w);  # x->(@info agent_count(w)) is not called
logcb(w);  # x->(@info agent_count(w)) *is* called
```
```@raw html
</div></div>
```
