### old definition

#=
# animals
abstract type Sheep <: AnimalSpecies end
abstract type Wolf <: AnimalSpecies end

# plants
abstract type Mushroom <: PlantSpecies end
abstract type Grass <: PlantSpecies end

export Grass, Sheep, Wolf, Mushroom

Base.show(io::IO, ::Type{Sheep}) = print(io,"🐑")
Base.show(io::IO, ::Type{Wolf}) = print(io,"🐺")

Base.show(io::IO,::Type{Mushroom}) = print(io,"🍄")
Base.show(io::IO, ::Type{Grass}) = print(io,"🌿")


function EcosystemCore.eat!(s::Animal{Sheep}, m::Plant{Mushroom}, w::World)
    if size(p)>0
        incr_energy!(s, -size(m)*Δenergy(s))
        m.size = 0
    end
end

function EcosystemCore.eat!(a::Animal{Wolf}, b::Animal{Sheep}, w::World)
    incr_energy!(a, energy(b)*Δenergy(a))
    kill_agent!(b,w)
end

function EcosystemCore.eat!(a::Animal{Sheep}, b::Plant{Grass}, w::World)
    incr_energy!(a, size(b)*Δenergy(a))
    b.size = 0
end

EcosystemCore.eats(::Animal{Sheep}, ::Plant{Mushroom}) = true
EcosystemCore.eats(::Animal{Sheep}, ::Plant{Grass}) = true
EcosystemCore.eats(::Animal{Wolf},::Animal{Sheep}) = true

EcosystemCore.mates(::Animal{S,Female}, ::Animal{S,Male}) where S<:Species = true
EcosystemCore.mates(::Animal{S,Male}, ::Animal{S,Female}) where S<:Species = true
EcosystemCore.mates(a::Agent, b::Agent) = false
=#

### new definition using macros from `ecosystem_macros.jl`
@plant  Grass       🌿
@plant  Broccoli    🥦
@plant  Mushroom    🍄
@animal Sheep       🐑
@animal Wolf        🐺
@animal Rabbit      🐇

@eats Rabbit [Grass => 0.5, Broccoli => 1.0]
@eats Sheep  [Grass => 0.5, Broccoli => 1.0, Mushroom => -1.0]
@eats Wolf   [Sheep => 0.9]