# Ecosystem debugging
Let's now apply what we have learned so far on the much bigger codebase of our `Ecosystem` and `EcosystemCore` packages. 

!!! note "Installation of Ecosystem pkg"
    If you do not have Ecosystem readily available you can get it from our [repository](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/master/src/Ecosystem.jl).

```@example block
include("ecosystems/lab04/Ecosystem.jl")

function make_counter()
    n = 0
    counter() = n += 1
end

function create_world()
    n_grass  = 1_000
    n_sheep  = 40
    n_wolves = 4

    nextid = make_counter()

    World(vcat(
        [Grass(nextid()) for _ in 1:n_grass],
        [Sheep(nextid()) for _ in 1:n_sheep],
        [Wolf(nextid()) for _ in 1:n_wolves],
    ))
end
world = create_world();
nothing # hide
```

Precompile everything by running one step of our simulation and run the profiler.

```julia
world_step!(world)
@profview for i=1:100 world_step!(world) end
```
![lab04-ecosystem](ecosystems/lab04-worldstep.png)

Red bars indicate type instabilities. The bars stacked on top of them are high,
narrow and not filling the whole width, indicating that the problem is pretty
serious. In our case the worst offender is the `filter` method inside
`find_food` and `find_mate` functions.
In both cases the bars on top of it are narrow and not the full with, meaning
that not that much time has been really spend working, but instead inferring the
types in the function itself during runtime.

As a reminder, this is the `find_food` function:
```julia
# original
function find_food(a::Animal, w::World)
    as = filter(x -> eats(a,x), w.agents |> values |> collect)
    isempty(as) ? nothing : sample(as)
end
```
Just from looking at that piece of code its not obvious what is the problem,
however the red color indicates that the code may be type unstable. Let's see if
that is the case by evaluation the function with some isolated inputs.

```@example block
using InteractiveUtils # hide
w = Wolf(4000)
find_food(w, world)
@code_warntype find_food(w, world)
```

Indeed we see that the return type is not inferred precisely but ends up being
just the `Union{Nothing, Agent}`, this is better than straight out `Any`, which
is the union of all types but still, julia has to do dynamic dispatch here, which is slow.

The underlying issue here is that we are working array of type `Vector{Agent}`,
where `Agent` is abstract, which does not allow the compiler to specialize the
code for the loop body.


# Different `Ecosystem.jl` versions

In order to fix the type instability in the `Vector{Agent}` we somehow have to
rethink our world such that we get a concrete type. Optimally we would have one
vector for each type of agent that populates our world. Before we completely
redesign how our world works we can try a simple hack that might already improve
things. Instead of letting julia figure our which types of agents we have (which
could be infinitely many), we can tell the compiler at least that we have only
three of them: `Wolf`, `Sheep`, and `Grass`.

We can do this with a tiny change in the constructor of our `World`:

```julia
function World(agents::Vector{<:Agent})
    ids = [a.id for a in agents]
    length(unique(ids)) == length(agents) || error("Not all agents have unique IDs!")

    # construct Dict{Int,Union{Animal{Wolf}, Animal{Sheep}, Plant{Grass}}}
    # instead of Dict{Int,Agent}
    types = unique(typeof.(agents))
    dict = Dict{Int,Union{types...}}(a.id => a for a in agents)

    World(dict, maximum(ids))
end
```

It turns out that with this simple change we can already gain a little bit of speed:

|              | normal     | `Animal{A}` & `Union` |
|--------------|------------|-----------------------|
| `find_food`  | 43.917 μs  | 12.208 μs             |
| `reproduce!` | 439.666 μs | 340.041 μs            |

The scripts to produce the numbers above can be found
[here](ecosystems/lab04/bench.jl) for our original version and
[here](ecosystems/animal_S_world_DictUnion/bench.jl) for the `Dict{Int,Union{...}}` version.

This however, does not yet fix our type instabilities completely. We are still working with `Union`s of types.
Next we could create a world that - instead of one plain dictionary - works with a tuple of dictionaries
with one entry for each type of agent. Our world would then look like this:
```julia
# pseudocode:
world ≈ (
    :Grass = Dict{Int, Plant{Grass}}(...),
    :Sheep = Dict{Int, Animal{Sheep}}(...),
    :Wolf = Dict{Int, Animal{Wolf}}(...)
)
```
In order to make this work we have to touch our ecosystem code in a number of
places, mostly related to `find_food` and `reproduce!`.  You can find a working
version of the ecosystem with a world based on `NamedTuple`s
[here](ecosystems/animal_S_world_NamedTupleDict/Ecosystem.jl).
With this slightly more involved update we can gain another bit of speed:

|              | normal     | `Animal{A}` & `Union` | `Animal{A}` & `NamedTuple` |
|--------------|------------|-----------------------|----------------------------|
| `find_food`  | 43.917 μs  | 12.208 μs             | 8.639 μs                   |
| `reproduce!` | 439.666 μs | 340.041 μs            | 273.103 μs                 |


The last optimization we can do is to move the `Sex` of our animals from a field
into a parametric type. Our world would then look like below:
```julia
# pseudocode:
world ≈ (
    :Grass = Dict{Int, Plant{Grass}}(...),
    :Sheep = Dict{Int, Animal{Sheep,Female}}(...),
    :Sheep = Dict{Int, Animal{Sheep,Male}}(...),
    :Wolf = Dict{Int, Animal{Wolf,Female}}(...)
    :Wolf = Dict{Int, Animal{Wolf,Male}}(...)
)
```
This should give us a lot of speedup in the `reproduce!` function, because we
will not have to `filter` for the correct sex anymore, but instead can just pick
the `NamedTuple` that is associated with the correct type of mate.
Unfortunately, changing the type signature of `Animal` essentially means that we
have to touch every line of code of our original ecosystem. However, the gain we
get for it is quite significant:

|              | normal     | `Animal{A}` & `Union` | `Animal{A}` & `NamedTuple` | `Animal{A,S}` & `NamedTuple` |
|--------------|------------|-----------------------|----------------------------|------------------------------|
| `find_food`  | 43.917 μs  | 12.208 μs             | 8.639 μs                   | 7.823 μs                     |
| `reproduce!` | 439.666 μs | 340.041 μs            | 273.103 μs                 | 77.646 ns                    |

The implementation of the new version with two parametric types can be found [here](ecosystems/animal_ST_world_NamedTupleDict/Ecosystem.jl). The completely blue (i.e. type stable) `@profview` of this version of the Ecosystem is quite satisfying to see

![neweco](ecosystems/animal_ST_world_NamedTuple_worldstep.png)

The same is true for the output of `@code_warntype`

```@example
using InteractiveUtils # hide
include("ecosystems/animal_ST_world_NamedTupleDict/Ecosystem.jl")

function make_counter()
    n = 0
    counter() = n += 1
end

function create_world()
    n_grass  = 1_000
    n_sheep  = 40
    n_wolves = 4

    nextid = make_counter()

    World(vcat(
        [Grass(nextid()) for _ in 1:n_grass],
        [Sheep(nextid()) for _ in 1:n_sheep],
        [Wolf(nextid()) for _ in 1:n_wolves],
    ))
end
world = create_world();

w = Wolf(4000)
find_food(w, world)
@code_warntype find_food(w, world)
```
