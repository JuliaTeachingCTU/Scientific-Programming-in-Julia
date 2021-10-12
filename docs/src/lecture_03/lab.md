# [Lab 3: Predator-Prey Agents](@id lab03)

In this lab we will look at two different ways of extending our agent
simulation to take into account that animals can have two different sexes:
*female* and *male*.

In the first part of the lab you will re-use the code from [lab 2](@ref lab02)
and create a new type of sheep (`⚥Sheep`) which has an additional field *sex*.
In the second part you will redesign the type hierarchy from scratch using
parametric types to make this agent system much more flexible and *julian*.

## Part I: Female & Male Sheep

The code from lab 2 that you will need in the first part of this lab can be
found [here](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/master/docs/src/lecture_02/Lab02Ecosystem.jl).

The goal of the first part of the lab is to demonstrate the *forwarding method*
(which is close to how things are done in OOP) by implementing a sheep that can
have two different sexes and can only reproduce with another sheep of opposite sex.

This new type of sheep needs an additonal field `sex::Symbol` which can be either
`:male` or `:female`.
In OOP we would simply inherit from `Sheep` and create a `⚥Sheep`
with an additional field. In Julia there is no inheritance - only subtyping of
abstract types.
As you cannot inherit from a concrete type in Julia, we will have to create a
wrapper type and forward all necessary methods. This is typically a sign of
unfortunate type tree design and should be avoided, but if you want to extend a
code base by an unforeseen type this forwarding of methods is a nice
work-around.  Our `⚥Sheep` type will simply contain a classic `sheep` and a
`sex` field
```@example lab03-nonparametric
include("../lecture_02/Lab02Ecosystem.jl") # hide
struct ⚥Sheep <: Animal
    sheep::Sheep
    sex::Symbol
end
⚥Sheep(id,E,ΔE,pr,pf,sex) = ⚥Sheep(Sheep(id,E,ΔE,pr,pf),sex)
nothing # hide
```

```@repl lab03-nonparametric
⚥Sheep(1,1.0,1.0,1.0,1.0,:female)
```

In our case, the methods that have to be forwarded are the accessors,
`eats` and `eat!`.  The custom reproduction behaviour will of
course be taken care of by a `reproduce!` function that does not just
forward but also contains specialized behaviour for the `⚥Sheep`.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Forward the accessors `energy`, `energy!`, `reprprob`, and `foodprob`,
as well as our core methods `eats` and `eat!` to `Sheep`.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab03-nonparametric
id(g::⚥Sheep) = id(g.sheep)
energy(g::⚥Sheep) = energy(g.sheep)
energy!(g::⚥Sheep, ΔE) = energy!(g.sheep, ΔE)
reprprob(g::⚥Sheep) = reprprob(g.sheep)
foodprob(g::⚥Sheep) = foodprob(g.sheep)

eats(::⚥Sheep, ::Grass) = true
eat!(s::⚥Sheep, g::Plant, w::World) = eat!(s.sheep, g, w)
nothing # hide
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
```@example lab03-nonparametric
mates(a::Plant, ::⚥Sheep) = false
mates(a::Animal, ::⚥Sheep) = false
mates(g1::⚥Sheep, g2::⚥Sheep) = g1.sex != g2.sex
function find_mate(g::⚥Sheep, w::World)
    ms = filter(a->mates(a,g), w.agents |> values |> collect)
    isempty(ms) ? nothing : sample(ms)
end

function reproduce!(s::⚥Sheep, w::World)
    m = find_mate(s,w)
    if !isnothing(m)
        energy!(s, energy(s)/2)
        vals = [getproperty(s.sheep,n) for n in fieldnames(Sheep) if n!=:id]
        new_id = w.max_id + 1
        ŝ = ⚥Sheep(new_id, vals..., rand(Bool) ? :female : :male)
        w.agents[id(ŝ)] = ŝ
        w.max_id = new_id
    end
end
nothing # hide
```
```@raw html
</p></details>
```

```@example lab03-nonparametric
f = ⚥Sheep(1,3.0,1.0,1.0,1.0,:female)
m = ⚥Sheep(2,4.0,1.0,1.0,1.0,:male)
w = World([f,m])

for _ in 1:4
    @show w
    world_step!(w)
end
```


## Part II: A new, parametric type hierarchy

You may have thought that the extention of Part I is not the most elegant thing
you have done in your life. If you did - you were right. There is a way of using
Julia's powerful type system to create a much more general verion of our agent
simulation. First, let us not that there are two fundamentally different types
of agents in our world: animals and plants. All species such as grass, sheep, wolves, etc.
can be categorized as on of those two.
Second, animals have two different, immutable sexes.  Thus an animal is
specified by two things: its *species* and its *sex*.  With this observation
let's try to redesign the type hiearchy using parametric types to reflect this.

The goal will be a `Plant` type with two parametric types: A `Species` type and
a `Sex` type. The type of a female wolf would then be `Animal{Wolf,Female}`.
The new type hiearchy then boils down to
```@example lab03
abstract type Species end
abstract type PlantSpecies <: Species end
abstract type Grass <: PlantSpecies end

abstract type AnimalSpecies <: Species end
abstract type Sheep <: AnimalSpecies end
abstract type Wolf <: AnimalSpecies end

abstract type Sex end
abstract type Male <: Sex end
abstract type Female <: Sex end

abstract type Agent{S<:Species} end
```
Now we can create a *concrete* type `Animal` with the two parametric types
and the fields that we already know from lab 2.
```@example lab03
mutable struct Animal{A<:AnimalSpecies,S<:Sex} <: Agent{A}
    id::Int
    energy::Float64
    Δenergy::Float64
    reprprob::Float64
    foodprob::Float64
end

# the accessors from lab 2 stay the same
id(a::Agent) = a.id
energy(a::Animal) = a.energy
Δenergy(a::Animal) = a.Δenergy
reprprob(a::Animal) = a.reprprob
foodprob(a::Animal) = a.foodprob
energy!(a::Animal, e) = a.energy = e
incr_energy!(a::Animal, Δe) = energy!(a, energy(a)+Δe)
nothing # hide
```
To create an instance of `Animal` we have to specify the parametric type
while constructing it
```@example lab03
Animal{Wolf,Female}(1,5,5,1,1)
```
Note that we now automatically have animals of any sex without additional work.
As a little enjoyable side project, we can overload Julia's `show` method to
get custom printing behaviour of our shiny new parametric type:
```@example lab03
Base.show(io::IO, ::Type{Sheep}) = print(io,"🐑")
Base.show(io::IO, ::Type{Wolf}) = print(io,"🐺")
Base.show(io::IO, ::Type{Male}) = print(io,"♂")
Base.show(io::IO, ::Type{Female}) = print(io,"♀")
function Base.show(io::IO, a::Animal{A,S}) where {A,S}
    e = energy(a)
    d = Δenergy(a)
    pr = reprprob(a)
    pf = foodprob(a)
    print(io,"$A$S #$(id(a)) E=$e ΔE=$d pr=$pr pf=$pf")
end

[Animal{Sheep,Male}(2,2,2,1,1),Animal{Wolf,Female}(1,5,5,1,1)]
```
Unfortunately we have lost the convenience of creating plants and animals
by simply calling their species constructor. For example, `Sheep` is just an
abstract type that we cannot instantiate. However, we can manually define
a new constructor that will give us this convenience back.
This is done in exactly the same way as defining a constructor for a concrete type
(i.e. turning it into a [function-like object](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects-1):
```julia
Sheep(id,E,ΔE,pr,pf,S=rand(Bool) ? Female : Male) = Animal{Sheep,S}(id,E,ΔE,pr,pf)
```
Ok, so we have a constructor for `Sheep` now. But what about all the other
billions of species that I want to define in my huge master project of
ecosystem simulations?  Do I have to write them all by hand? *Do not
despair!* Julia has you covered:
```@example lab03
function (A::Type{<:AnimalSpecies})(id,E,ΔE,pr,pf,S=rand(Bool) ? Female : Male)
    Animal{A,S}(id,E,ΔE,pr,pf)
end

[Sheep(2,2,2,1,1),Wolf(1,5,5,1,1)]
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Adapt the code from lab 2 to work with our new parametric type hierarchy.
For this you will have to define a concrete `Plant` type in a similar fashion
as the new `Animal` type. Additionally you need to adapt at least the methods
`eat!`, `eats`, `mates`, and `reproduce!`.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
The full solution can be found on Github for the
[`World`](https://github.com/JuliaTeachingCTU/EcosystemCore.jl/blob/main/src/world.jl),
[`Plant`s](https://github.com/JuliaTeachingCTU/EcosystemCore.jl/blob/main/src/plant.jl), and
[`Animal`s](https://github.com/JuliaTeachingCTU/EcosystemCore.jl/blob/main/src/animal.jl).
```@raw html
</p></details>
```
