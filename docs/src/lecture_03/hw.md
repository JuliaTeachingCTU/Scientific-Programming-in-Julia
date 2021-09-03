# Homework 3

```@setup load_ecosystem
using Scientific_Programming_in_Julia
using Scientific_Programming_in_Julia.Ecosystem: eat!, find_food, count
```

The goal of this homework is to demonstrate the [forwarding method](@ref forwarding_method) by
implementing a gendered sheep that can only reproduce with another sheep of
opposite gender.

The gendered sheep need an additonal field `gender::Symbol` which can be either
`:male` or `:female`.
In OOP we would now simply inherit from `Sheep` and create a `GenderedSheep`
with an additional field. In Julia there is no inheritance - only subtyping of
abstract types.
As you cannot inherit from a concrete type in Julia, we will have to create a
wrapper type and forward all necessary methods. This is typically a sign of
unfortunate type tree design and should be avoided, but if you want to extend a
code base by a type that was not thought of during the inital design, this
forwarding of methods is a nice work-around.  Our `GenderedSheep` type will
simply contain a classic `sheep` and a `gender` field
```julia
struct GenderedSheep{T<:Real} <: AbstractAnimal
    sheep::Sheep{T}
    gender::Symbol
end
GenderedSheep(E,ΔE,pr,pf,gender) = GenderedSheep(Sheep(E,ΔE,pr,pf),gender)
```

```@repl load_ecosystem
GenderedSheep(1.0,1.0,1.0,1.0,:female)
```

In our case, the methods that have to be forwarded are `agent_step!`,
`reproduce!`, `eats` and `eat!`.  The custom reproduction behaviour will of
course be taken care of by a `reproduce!` function that does not just
forward but also contains specialized behaviour for the `GenderedSheep`.


```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework</header>
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
energy(g::GenderedSheep) = energy(g.sheep)
energy!(g::GenderedSheep, ΔE) = energy!(g.sheep, ΔE)
reproduction_prob(g::GenderedSheep) = reproduction_prob(g.sheep)
food_prob(g::GenderedSheep) = food_prob(g.sheep)

eats(::GenderedSheep, ::Grass) = true
eats(::GenderedSheep, ::PoisonedGrass) = true
eat!(s::GenderedSheep, g::AbstractPlant, w::World) = eat!(s.sheep, g, w)
```
```@raw html
</p></details>
```

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework</header>
<div class="admonition-body">
```
Implement the `reproduce!` method for the `GenderedSheep`.  Note that you first
have to find another sheep of opposite gender in your `World`, and only if you
can find one you can reproduce.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```julia
mates(a::AbstractPlant, ::GenderedSheep) = false
mates(a::AbstractAnimal, ::GenderedSheep) = false
mates(g1::GenderedSheep, g2::GenderedSheep) = g1.gender != g2.gender
function find_mate(g::GenderedSheep, w::World)
    ms = filter(a->mates(a,g), w.agents)
    isempty(ms) ? nothing : sample(ms)
end

function reproduce!(s::GenderedSheep, w::World)
    m = find_mate(s,w)
    if !isnothing(m)
        s.sheep.energy /= 2
        # TODO: should probably mix s/m
        push!(w.agents, deepcopy(s))
    end
end
```
```@raw html
</p></details>
```
