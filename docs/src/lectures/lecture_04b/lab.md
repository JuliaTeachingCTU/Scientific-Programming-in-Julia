# Lab 04b: Interfaces, traits and checking the Ecosystem

```@setup block
using StatsBase
include("../lecture_03/Lab03Ecosystem.jl")

function find_food(a::Animal, w::World)
    as = filter(x -> eats(a,x), w.agents |> values |> collect)
    isempty(as) ? nothing : sample(as)
end

eats(::Animal{Sheep},g::Plant{Grass}) = g.size > 0
eats(::Animal{Wolf},::Animal{Sheep}) = true
eats(::Agent,::Agent) = false
```

## Warmup - Stepping through time

We now have all necessary functions in place to make agents perform one step
of our simulation.  At the beginning of each step an animal looses energy.
Afterwards it tries to find some food, which it will subsequently eat. If the
animal then has less than zero energy it dies and is removed from the world. If
it has positive energy it will try to reproduce.

Plants have a simpler life. They simply grow if they have not reached their maximal size.

::: warning Exercise

1. Implement a method `agent_step!(::Animal,::World)` which performs the following steps:
    - Decrement $E$ of agent by `1.0`.
    - With $p_f$, try to find some food and eat it.
    - If $E<0$, the animal dies.
    - With $p_r$, try to reproduce.
2. Implement a method `agent_step!(::Plant,::World)` which performs the following steps:
    - If the size of the plant is smaller than `max_size`, increment the plant's size by one.

:::

::: details Show solution

```@example block
function agent_step!(p::Plant, w::World)
    if p.size < p.max_size
        p.size += 1
    end
end

function agent_step!(a::Animal, w::World)
    a.energy -= 1
    if rand() <= a.foodprob
        dinner = find_food(a,w)
        eat!(a, dinner, w)
    end
    if a.energy < 0
        kill_agent!(a,w)
        return
    end
    if rand() <= a.reprprob
        reproduce!(a,w)
    end
end

nothing # hide
```

:::

An `agent_step!` of a sheep in a world with a single grass should make it consume the grass,
let it reproduce, and eventually die if there is no more food and its energy is at zero:

```@repl block
sheep = Sheep(1,2.0,2.0,1.0,1.0,male);
grass = Grass(2,2,2);
world = World([sheep, grass])
agent_step!(sheep, world); world
# NOTE: The second agent step leads to an error.
# Can you figure out what is the problem here?
agent_step!(sheep, world); world
```

::: warning Exercise

Finally, lets implement a function `world_step!` which performs one
`agent_step!` for each agent.  Note that simply iterating over all agents could
lead to problems because we are mutating the agent dictionary.  One solution for
this is to iterate over a copy of all agent IDs that are present when starting
to iterate over agents.  Additionally, it could happen that an agent is killed
by another one before we apply `agent_step!` to it. To solve this you can check
if a given ID is currently present in the `World`.

:::

::: details Show solution

```@example block
# make it possible to eat nothing
eat!(::Animal, ::Nothing, ::World) = nothing

function world_step!(world::World)
    # make sure that we only iterate over IDs that already exist in the
    # current timestep this lets us safely add agents
    ids = copy(keys(world.agents))

    for id in ids
        # agents can be killed by other agents, so make sure that we are
        # not stepping dead agents forward
        !haskey(world.agents,id) && continue

        a = world.agents[id]
        agent_step!(a,world)
    end
end

nothing #hide
```

:::

```@repl block
w = World([Sheep(1), Sheep(2), Wolf(3)])
world_step!(w); w
world_step!(w); w
world_step!(w); w
```

Finally, lets run a few simulation steps and plot the solution

```@example block
n_grass  = 1_000
n_sheep  = 40
n_wolves = 4

gs = [Grass(id) for id in 1:n_grass]
ss = [Sheep(id) for id in (n_grass+1):(n_grass+n_sheep)]
ws = [Wolf(id) for id in (n_grass+n_sheep+1):(n_grass+n_sheep+n_wolves)]
w  = World(vcat(gs,ss,ws))

counts = Dict(n=>[c] for (n,c) in agent_count(w))
for _ in 1:100
    world_step!(w)
    for (n,c) in agent_count(w)
        push!(counts[n],c)
    end
end

using Plots

plt = plot()
for (n,c) in counts
    plot!(plt, c, label=string(n), lw=2)
end
plt
```

## Package: `Ecosystem.jl`

You know how to create a package from the bachelor course. We will need it in the following labs.

::: warning Exercise

1. Create a package `Ecosystem.jl` (`PkgTemplates.jl` or `]generate Ecosystem`).
2. Copy the code of the ecosystem including the stepping functions from the warmup into `src/Ecosystem.jl`.
3. Add `StatsBase` as a dependency and export the types and functions.
4. Check that `using Ecosystem` and `world_step!` work in a fresh session (`Revise.jl` recommended).

:::

::: details Show solution

See [lab 4 of 2025](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/2025W/docs/src/lectures/lecture_04/lab.md) for a step-by-step guide. The resulting source file is [Lab04Ecosystem.jl](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/2025W/docs/src/lectures/lecture_04/Lab04Ecosystem.jl) wrapped in

```julia
module Ecosystem

using StatsBase: sample

# ... content of Lab04Ecosystem.jl ...

export World, Species, PlantSpecies, AnimalSpecies, Grass, Sheep, Wolf
export Agent, Animal, Plant, Sex, female, male
export agent_step!, world_step!, agent_count, eats, eat!, find_food, reproduce!, kill_agent!

end
```

:::

## Part I: The World as a collection

So far, every function digs into `w.agents |> values |> collect`. The `World` is a collection of agents, let it behave like one.

::: warning Exercise

Implement the [iteration interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration) for `World`, iterating over its agents:

- `Base.iterate`
- `Base.length`
- `Base.eltype`

Then check what you got for free:

```julia
w = World([Grass(1), Sheep(2), Wolf(3)])
for a in w
    println(a)
end
collect(w)
count(a -> a isa Animal{Sheep}, w)
sum(agent_count, w)
filter(a -> eats(Wolf(4), a), collect(w))
```

**HINTS**:
- The agents are stored in a `Dict`, which is already iterable. Can you forward the iteration to `values(w.agents)`?
- `eltype` is defined on the *type*: `Base.eltype(::Type{World{A}}) where A`.

:::

::: details Show solution

```julia
Base.iterate(w::World, s...) = iterate(values(w.agents), s...)
Base.length(w::World) = length(w.agents)
Base.eltype(::Type{World{A}}) where A = A
```

Three lines and the `World` works with `for`, `collect`, `count`, `sum`, `filter`, `map`, ...

:::

::: tip Discussion

- What happens with `collect(w)` if you do not define `length`? Which trait is responsible? (See the lecture.)
- Rewrite `find_food` using the new interface. Is it simpler?
- Is it safe to iterate over the `World` while `world_step!` adds and removes agents?

:::

## Part II: Who eats whom? A `Diet` trait

The current rules are written species by species:

```julia
eats(::Animal{Sheep}, g::Plant{Grass}) = g.size > 0
eats(::Animal{Wolf}, ::Animal{Sheep}) = true
eats(::Agent, ::Agent) = false
```

and similarly for `eat!`. Adding a rabbit means writing new methods for rabbits eating grass and for wolves eating rabbits. Adding a fox ... The table grows with the square of the number of species.

What really matters is the *diet* of the species: herbivores eat plants, carnivores eat herbivores.

::: warning Exercise

1. Define a trait `Diet` with values `Herbivore` and `Carnivore`.
2. Assign `Diet` to `Sheep` and `Wolf`.
3. Replace the species-specific methods of `eats` and `eat!` by methods dispatching on the trait:
   - a herbivore eats a plant with `size > 0`,
   - a carnivore eats an animal whose species is a herbivore.
4. Check that the simulation behaves as before.

**HINTS**:
- Trait function on the species type: `Diet(::Type{Sheep}) = Herbivore()`.
- Trait dispatch: `eats(a::Animal{A}, b::Agent) where A = eats(Diet(A), b)`.
- Keep the fallback for anything that is not food.

:::

::: details Show solution

```julia
abstract type Diet end
struct Herbivore <: Diet end
struct Carnivore <: Diet end

Diet(::Type{Sheep}) = Herbivore()
Diet(::Type{Wolf}) = Carnivore()

eats(a::Animal{A}, b::Agent) where A = eats(Diet(A), b)
eats(::Herbivore, p::Plant) = p.size > 0
eats(::Carnivore, b::Animal{B}) where B = Diet(B) isa Herbivore
eats(::Diet, ::Agent) = false

eat!(a::Animal{A}, b::Agent, w::World) where A = eat!(Diet(A), a, b, w)
function eat!(::Herbivore, a::Animal, p::Plant, ::World)
    a.energy += p.size * a.Δenergy
    p.size = 0
end
function eat!(::Carnivore, a::Animal, b::Animal, w::World)
    a.energy += b.energy * a.Δenergy
    kill_agent!(b, w)
end
```

Note that `eat!(::Animal, ::Nothing, ::World)` stays, `nothing` is not an `Agent`.

:::

The payoff: a new species `Rabbit`, defined *outside* of the package, without modifying `eats` or `eat!`:

```julia
using Ecosystem

abstract type Rabbit <: AnimalSpecies end
Rabbit(id; E=2.0, ΔE=0.3, pr=0.9, pf=0.7, s=rand(instances(Sex))) = Rabbit(id, E, ΔE, pr, pf, s)
Base.show(io::IO, ::Type{Rabbit}) = print(io, "🐇")

Ecosystem.Diet(::Type{Rabbit}) = Herbivore()   # the only line about food

eats(Rabbit(1), Grass(2, 3, 3))   # true
eats(Wolf(1), Rabbit(2))          # true
```

Rabbits eat grass and wolves eat rabbits. Neither `eats` nor `eat!` was touched.

::: tip Discussion

- Why not `abstract type Herbivore <: AnimalSpecies end` and `abstract type Sheep <: Herbivore end`? What if we want to split species also by habitat or by size?
- Is `Ecosystem.Diet(::Type{Rabbit})` type piracy?
- Wolves eating rabbits more eagerly than sheep: can a specific method `eats(::Animal{Wolf}, ::Animal{Rabbit})` override the trait?

:::

## Part III: Checking the Ecosystem

Nothing of the above is checked by the compiler. Let us bring some rules back.

::: warning Exercise

Add tests to `test/runtests.jl`:

1. A testset for the iteration interface of `World` (`length`, `collect`, `eltype`).
2. A testset for the `Diet` trait: who eats whom (wolf→sheep, sheep→grass, *not* sheep→wolf, *not* wolf→wolf).

Run them with `]test`.

:::

Automated checks of the package with [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) are part of the [homework](hw.md).

::: warning Exercise

A colleague contributed a function to the package:

```julia
total_energy(w::World) = sum(a.enrgy for a in w if a isa Animal)
```

Use [JET.jl](https://github.com/aviatesk/JET.jl) to find the problems *without* running the simulation:

```julia
using JET
w = World([Grass(1), Sheep(2)])
@report_call total_energy(w)
```

1. Fix the reported problems.
2. Change the function to `sum(a.energy for a in w)` (all agents). What does JET say for `World([Grass(1), Sheep(2)])`, and what for `World([Grass(1), Grass(2)])`? Why the difference?
3. Run `report_package(Ecosystem)`.

:::

::: details Show solution

JET reports two problems:

```
FieldError: type Animal has no field `enrgy`, available fields: `id`, `energy`, `Δenergy`, `reprprob`, `foodprob`, `sex`
may throw: ArgumentError("reducing over an empty collection is not allowed; consider supplying `init` to the reducer")
```

```julia
total_energy(w::World) = sum((a.energy for a in w if a isa Animal); init=0.0)
```

On `World{Agent}`, JET cannot prove that a `Plant` will ever reach `a.energy`, the element type is abstract. On `World{Plant{Grass}}` it can, and reports `FieldError: type Plant has no field energy`. JET sees only what the types tell it.

:::

::: tip Outlook

Try `@report_opt world_step!(w)`. JET reports *runtime dispatch*. What is it and does it matter? Next lecture.

:::
