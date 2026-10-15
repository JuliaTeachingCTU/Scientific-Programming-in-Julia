# Ecosystem package module.
module Ecosystem

using Random: rand

export World, Species, PlantSpecies, AnimalSpecies, Grass, Sheep, Wolf
export Agent, Animal, Plant, Sex, female, male
export Diet, Herbivore, Carnivore
export agent_step!, world_step!, agent_count, eats, eat!, find_food, reproduce!, kill_agent!
export total_energy

abstract type Species end
abstract type PlantSpecies <: Species end
abstract type Grass <: PlantSpecies end
abstract type AnimalSpecies <: Species end
abstract type Sheep <: AnimalSpecies end
abstract type Wolf <: AnimalSpecies end

abstract type Agent{S<:Species} end

@enum Sex female male

mutable struct World{A<:Agent}
    agents::Dict{Int,A}
    max_id::Int
end

function World(agents::AbstractVector{<:Agent})
    A = eltype(agents)
    World{A}(Dict{Int,A}(a.id => a for a in agents), isempty(agents) ? 0 : maximum(a.id for a in agents))
end

Base.iterate(w::World, state...) = iterate(values(w.agents), state...)
Base.length(w::World) = length(w.agents)
Base.eltype(::Type{World{A}}) where {A} = A

function Base.show(io::IO, w::World)
    println(io, typeof(w))
    for a in w
        println(io, "  ", a)
    end
end

mutable struct Animal{A<:AnimalSpecies} <: Agent{A}
    const id::Int
    energy::Float64
    const Δenergy::Float64
    const reprprob::Float64
    const foodprob::Float64
    const sex::Sex
end

function (A::Type{<:AnimalSpecies})(id::Int, E::Real, ΔE::Real, pr::Real, pf::Real, s::Sex)
    Animal{A}(id, E, ΔE, pr, pf, s)
end

randsex() = rand(instances(Sex))
Sheep(id; E=4.0, ΔE=0.2, pr=0.8, pf=0.6, s=randsex()) = Sheep(id, E, ΔE, pr, pf, s)
Wolf(id; E=10.0, ΔE=8.0, pr=0.1, pf=0.2, s=randsex()) = Wolf(id, E, ΔE, pr, pf, s)

function Base.show(io::IO, a::Animal{A}) where {A<:AnimalSpecies}
    sex = a.sex == female ? "♀" : "♂"
    print(io, "$A$sex #$(a.id) E=$(a.energy) ΔE=$(a.Δenergy) pr=$(a.reprprob) pf=$(a.foodprob)")
end
Base.show(io::IO, ::Type{Sheep}) = print(io, "🐑")
Base.show(io::IO, ::Type{Wolf}) = print(io, "🐺")

mutable struct Plant{P<:PlantSpecies} <: Agent{P}
    const id::Int
    size::Int
    const max_size::Int
end

(P::Type{<:PlantSpecies})(id::Int, size::Int, max_size::Int) = Plant{P}(id, size, max_size)
(P::Type{<:PlantSpecies})(id::Int, max_size::Int) = P(id, rand(1:max_size), max_size)
Grass(id; max_size=10) = Grass(id, rand(1:max_size), max_size)

function Base.show(io::IO, p::Plant{P}) where {P}
    print(io, "$P  #$(p.id) $(round(Int, 100p.size / p.max_size))% grown")
end
Base.show(io::IO, ::Type{Grass}) = print(io, "🌿")

abstract type Diet end
struct Herbivore <: Diet end
struct Carnivore <: Diet end

Diet(::Type{Sheep}) = Herbivore()
Diet(::Type{Wolf}) = Carnivore()

eats(a::Animal{A}, b::Agent) where {A} = eats(Diet(A), b)
eats(::Herbivore, p::Plant) = p.size > 0
eats(::Carnivore, b::Animal{B}) where {B} = Diet(B) isa Herbivore
eats(::Diet, ::Agent) = false

eat!(a::Animal{A}, b::Agent, w::World) where {A} = eat!(Diet(A), a, b, w)
function eat!(::Herbivore, a::Animal, p::Plant, ::World)
    a.energy += p.size * a.Δenergy
    p.size = 0
end
function eat!(::Carnivore, a::Animal, b::Animal, w::World)
    a.energy += b.energy * a.Δenergy
    kill_agent!(b, w)
end
eat!(::Animal, ::Nothing, ::World) = nothing

kill_agent!(a::Agent, w::World) = delete!(w.agents, a.id)

function find_food(a::Animal, w::World)
    food = filter(b -> eats(a, b), collect(w))
    isempty(food) ? nothing : rand(food)
end

mates(a::Animal{A}, b::Animal{A}) where {A<:AnimalSpecies} = a.sex != b.sex
mates(::Agent, ::Agent) = false

function reproduce!(a::Animal{A}, w::World) where {A}
    mates_for_a = filter(b -> mates(a, b), collect(w))
    isempty(mates_for_a) && return nothing
    a.energy /= 2
    id = w.max_id + 1
    child = Animal{A}(id, a.energy, a.Δenergy, a.reprprob, a.foodprob, randsex())
    w.agents[id] = child
    w.max_id = id
    nothing
end

function agent_step!(p::Plant, ::World)
    p.size < p.max_size && (p.size += 1)
    nothing
end

function agent_step!(a::Animal, w::World)
    a.energy -= 1
    rand() <= a.foodprob && eat!(a, find_food(a, w), w)
    a.energy < 0 && return kill_agent!(a, w)
    rand() <= a.reprprob && reproduce!(a, w)
    nothing
end

function world_step!(w::World)
    for id in copy(keys(w.agents))
        haskey(w.agents, id) && agent_step!(w.agents[id], w)
    end
    nothing
end

agent_count(p::Plant) = p.size / p.max_size
agent_count(::Animal) = 1
agent_count(agents) = sum(agent_count, agents)
function agent_count(w::World)
    counts = Dict{Symbol,Float64}()
    for a in w
        name = nameof(typeof(a).parameters[1])
        counts[name] = get(counts, name, 0.0) + agent_count(a)
    end
    counts
end

"""Return the total energy of all animals in `w`."""
total_energy(w::World) = sum((a.energy for a in w if a isa Animal); init=0.0)

end
