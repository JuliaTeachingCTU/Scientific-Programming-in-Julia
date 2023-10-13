# [Lab 3: Predator-Prey Agents](@id lab03)

```@setup forward
projdir = dirname(Base.active_project())
include(joinpath(projdir,"src","lecture_02","Lab02Ecosystem.jl"))
```

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
```@example forward
struct ⚥Sheep <: Animal
    sheep::Sheep
    sex::Symbol
end
⚥Sheep(id, e=4.0, Δe=0.2, pr=0.8, pf=0.6, sex=rand(Bool) ? :female : :male) = ⚥Sheep(Sheep(id,e,Δe,pr,pf),sex)
nothing # hide
```

```@repl forward
sheep = ⚥Sheep(1)
sheep.sheep
sheep.sex
```

Instead of littering the whole code with custom getters/setters Julia allows us
to overload the `sheep.field` behaviour by implementing custom
`getproperty`/`setproperty!` methods.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
Implement custom `getproperty`/`setproperty!` methods which allow to access the
`Sheep` inside the `⚥Sheep` as if we would not be wrapping it.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example forward
# NOTE: the @forward macro we will discuss in a later lecture is based on this

function Base.getproperty(s::⚥Sheep, name::Symbol)
    if name in fieldnames(Sheep)
        getfield(s.sheep,name)
    else
        getfield(s,name)
    end
end

function Base.setproperty!(s::⚥Sheep, name::Symbol, x)
    if name in fieldnames(Sheep)
        setfield!(s.sheep,name,x)
    else
        setfield!(s,name,x)
    end
end
```
```@raw html
</p></details>
```
You should be able to do the following with your overloads now
```@repl forward
sheep = ⚥Sheep(1)
sheep.id
sheep.sex
sheep.energy += 1
sheep
```

In order to make the `⚥Sheep` work with the rest of the code we only have
to forward the `eat!` method
```@repl forward
eat!(s::⚥Sheep, food, world) = eat!(s.sheep, food, world);
sheep = ⚥Sheep(1);
grass = Grass(2);
world = World([sheep,grass])
eat!(sheep, grass, world)
```
and implement a custom `reproduce!` method with the behaviour that we want.

However, the extension of `Sheep` to `⚥Sheep` is a very object-oriented approach.
With a little bit of rethinking, we can build a much more elegant solution that
makes use of Julia's powerful parametric types.


## Part II: A new, parametric type hierarchy

First, let us note that there are two fundamentally different types of agents in
our world: animals and plants. All species such as grass, sheep, wolves, etc.
can be categorized as one of those two.  We can use Julia's powerful,
*parametric* type system to define one large abstract type for all agents
`Agent{S}`. The `Agent` will either be an `Animal` or a `Plant` with a type
parameter `S` which will represent the specific animal/plant
species we are dealing with.

This new type hiearchy can then look like this:
```@example parametric
abstract type Species end

abstract type PlantSpecies <: Species end
abstract type Grass <: PlantSpecies end

abstract type AnimalSpecies <: Species end
abstract type Sheep <: AnimalSpecies end
abstract type Wolf <: AnimalSpecies end

abstract type Agent{S<:Species} end

# instead of Symbols we can use an Enum for the sex field
# using an Enum here makes things easier to extend in case you
# need more than just binary sexes and is also more explicit than
# just a boolean
@enum Sex female male
```
```@setup parametric
mutable struct World{A<:Agent}
    agents::Dict{Int,A}
    max_id::Int
end

function World(agents::Vector{<:Agent})
    max_id = maximum(a.id for a in agents)
    World(Dict(a.id=>a for a in agents), max_id)
end

# optional: overload Base.show
function Base.show(io::IO, w::World)
    println(io, typeof(w))
    for (_,a) in w.agents
        println(io,"  $a")
    end
end
```

Now we can create a *concrete* type `Animal` with the two parametric types
and the fields that we already know from lab 2.
```@example parametric
mutable struct Animal{A<:AnimalSpecies} <: Agent{A}
    const id::Int
    energy::Float64
    const Δenergy::Float64
    const reprprob::Float64
    const foodprob::Float64
    const sex::Sex
end
```
To create an instance of `Animal` we have to specify the parametric type
while constructing it
```@repl parametric
Animal{Wolf}(1,5,5,1,1,female)
```
Note that we now automatically have animals of any species without additional work.
Starting with the overload of the `show` method we can already see that we can
abstract away a lot of repetitive work into the type system. We can implement
*one single* `show` method for all animal species!
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
Implement `Base.show(io::IO, a::Animal)` with a single method for all `Animal`s.
You can get the pretty (unicode) printing of the `Species` types with
another overload like this: `Base.show(io::IO, ::Type{Sheep}) = print(io,"🐑")`
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example parametric
function Base.show(io::IO, a::Animal{A}) where {A<:AnimalSpecies}
    e = a.energy
    d = a.Δenergy
    pr = a.reprprob
    pf = a.foodprob
    s = a.sex == female ? "♀" : "♂"
    print(io, "$A$s #$(a.id) E=$e ΔE=$d pr=$pr pf=$pf")
end

# note that for new species/sexes we will only have to overload `show` on the
# abstract species types like below!
Base.show(io::IO, ::Type{Sheep}) = print(io,"🐑")
Base.show(io::IO, ::Type{Wolf}) = print(io,"🐺")
```
```@raw html
</p></details>
```

Unfortunately we have lost the convenience of creating plants and animals
by simply calling their species constructor. For example, `Sheep` is just an
abstract type that we cannot instantiate. However, we can manually define
a new constructor that will give us this convenience back.
This is done in exactly the same way as defining a constructor for a concrete type:
```julia
Sheep(id,E,ΔE,pr,pf,s=rand(Sex)) = Animal{Sheep}(id,E,ΔE,pr,pf,s)
```
Ok, so we have a constructor for `Sheep` now. But what about all the other
billions of species that you want to define in your huge master thesis project of
ecosystem simulations?  Do you have to write them all by hand? *Do not
despair!* Julia has you covered.
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
Overload all `AnimalSpecies` types with a constructor.
You already know how to write constructors for specific types such as `Sheep`.
Can you manage to sneak in a type variable? Maybe with `Type`?
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@example parametric
function (A::Type{<:AnimalSpecies})(id::Int,E::T,ΔE::T,pr::T,pf::T,s::Sex) where T
    Animal{A}(id,E,ΔE,pr,pf,s)
end

# get the per species defaults back
randsex() = rand(instances(Sex))
Sheep(id; E=4.0, ΔE=0.2, pr=0.8, pf=0.6, s=randsex()) = Sheep(id, E, ΔE, pr, pf, s)
Wolf(id; E=10.0, ΔE=8.0, pr=0.1, pf=0.2, s=randsex()) = Wolf(id, E, ΔE, pr, pf, s)
nothing # hide
```

```@raw html
</p></details>
```
We have our convenient, high-level behaviour back!
```@repl parametric
Sheep(1)
Wolf(2)
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
Check the methods for `eat!` and `kill_agent!` which involve `Animal`s and update
their type signatures such that they work for the new type hiearchy.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example parametric
function eat!(wolf::Animal{Wolf}, sheep::Animal{Sheep}, w::World)
    wolf.energy += sheep.energy * wolf.Δenergy
    kill_agent!(sheep,w)
end

# no change
# eat!(::Animal, ::Nothing, ::World) = nothing

# no change
# kill_agent!(a::Agent, w::World) = delete!(w.agents, a.id)

eats(::Animal{Wolf},::Animal{Sheep}) = true
eats(::Agent,::Agent) = false
# this one needs to wait until we have `Plant`s
# eats(::Animal{Sheep},g::Plant{Grass}) = g.size > 0

nothing # hide
```


```@raw html
</p></details>
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
Finally, we can implement the new behaviour for `reproduce!` which we wanted.
Build a function which first finds an animal species of opposite sex and then
lets the two reproduce (same behaviour as before).
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example parametric
mates(a::Animal{A}, b::Animal{A}) where A<:AnimalSpecies = a.sex != b.sex
mates(::Agent, ::Agent) = false

function find_mate(a::Animal, w::World)
    ms = filter(x->mates(x,a), w.agents |> values |> collect)
    isempty(ms) ? nothing : rand(ms)
end

function reproduce!(a::Animal{A}, w::World) where {A}
    m = find_mate(a,w)
    if !isnothing(m)
        a.energy = a.energy / 2
        vals = [getproperty(a,n) for n in fieldnames(Animal) if n ∉ [:id, :sex]]
        new_id = w.max_id + 1
        ŝ = Animal{A}(new_id, vals..., randsex())
        w.agents[ŝ.id] = ŝ
        w.max_id = new_id
    end
end
nothing # hide
```
```@raw html
</p></details>
```

```@repl parametric
s1 = Sheep(1, s=female)
s2 = Sheep(2, s=male)
w  = World([s1, s2])
reproduce!(s1, w); w
```


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
Implement the type hiearchy we designed for `Plant`s as well.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example parametric
mutable struct Plant{P<:PlantSpecies} <: Agent{P}
    id::Int
    size::Int
    max_size::Int
end

# constructor for all Plant{<:PlantSpecies} callable as PlantSpecies(...)
(A::Type{<:PlantSpecies})(id, s, m) = Plant{A}(id,s,m)
(A::Type{<:PlantSpecies})(id, m) = (A::Type{<:PlantSpecies})(id,rand(1:m),m)

# default specific for Grass
Grass(id; max_size=10) = Grass(id, rand(1:max_size), max_size)

function Base.show(io::IO, p::Plant{P}) where P
    x = p.size/p.max_size * 100
    print(io,"$P  #$(p.id) $(round(Int,x))% grown")
end

Base.show(io::IO, ::Type{Grass}) = print(io,"🌿")

function eat!(sheep::Animal{Sheep}, grass::Plant{Grass}, w::World)
    sheep.energy += grass.size * sheep.Δenergy
    grass.size = 0
end
eats(::Animal{Sheep},g::Plant{Grass}) = g.size > 0

nothing # hide
```
```@raw html
</p></details>
```

```@repl parametric
g = Grass(2)
s = Sheep(3)
w = World([g,s])
eat!(s,g,w); w
```
