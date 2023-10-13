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


mutable struct Animal{A<:AnimalSpecies} <: Agent{A}
    const id::Int
    energy::Float64
    const Δenergy::Float64
    const reprprob::Float64
    const foodprob::Float64
    const sex::Sex
end

function Base.show(io::IO, a::Animal{A}) where {A<:AnimalSpecies}
    e = a.energy
    d = a.Δenergy
    pr = a.reprprob
    pf = a.foodprob
    s = a.sex == female ? "♀" : "♂"
    print(io, "$A$s #$(a.id) E=$e ΔE=$d pr=$pr pf=$pf")
end

# note that for new species/sexes we will only have to overload `show` on the
# abstract species/sex types like below!
Base.show(io::IO, ::Type{Sheep}) = print(io,"🐑")
Base.show(io::IO, ::Type{Wolf}) = print(io,"🐺")
