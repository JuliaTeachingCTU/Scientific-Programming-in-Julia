using StatsBase

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

##########  World  #############################################################

mutable struct World{T<:NamedTuple}
    # this is a NamedTuple of Dict{Int,<:Agent}
    # but I don't know how to express that as a parametric type
    agents::T
    max_id::Int
end

function World(agents::Vector{<:Agent})
    types = unique(typeof.(agents))
    ags = map(types) do T
        as = filter(x -> isa(x,T), agents)
        Dict{Int,T}(a.id=>a for a in as)
    end
    nt = (; zip(tosym.(types), ags)...)
    
    ids = [a.id for a in agents]
    length(unique(ids)) == length(agents) || error("Not all agents have unique IDs!")
    World(nt, maximum(ids))
end

# optional: overload Base.show
function Base.show(io::IO, w::World)
    ts = join([valtype(a) for a in w.agents], ", ")
    println(io, "World[$ts]")
    for dict in w.agents
        for (_,a) in dict
            println(io,"  $a")
        end
    end
end

##########  Animals  ###########################################################

mutable struct Animal{A<:AnimalSpecies} <: Agent{A}
    const id::Int
    energy::Float64
    const Δenergy::Float64
    const reprprob::Float64
    const foodprob::Float64
    const sex::Sex
end

function (A::Type{<:AnimalSpecies})(id::Int,E::T,ΔE::T,pr::T,pf::T,s::Sex) where T
    Animal{A}(id,E,ΔE,pr,pf,s)
end

# get the per species defaults back
randsex() = rand(instances(Sex))
Sheep(id; E=4.0, ΔE=0.2, pr=0.8, pf=0.6, s=randsex()) = Sheep(id, E, ΔE, pr, pf, s)
Wolf(id; E=10.0, ΔE=8.0, pr=0.1, pf=0.2, s=randsex()) = Wolf(id, E, ΔE, pr, pf, s)


function Base.show(io::IO, a::Animal{A}) where {A<:AnimalSpecies}
    e = a.energy
    d = a.Δenergy
    pr = a.reprprob
    pf = a.foodprob
    s = a.sex == female ? "♀" : "♂"
    print(io, "$A$s #$(a.id) E=$e ΔE=$d pr=$pr pf=$pf")
end

# note that for new species we will only have to overload `show` on the
# abstract species/sex types like below!
Base.show(io::IO, ::Type{Sheep}) = print(io,"🐑")
Base.show(io::IO, ::Type{Wolf}) = print(io,"🐺")


##########  Plants  #############################################################

mutable struct Plant{P<:PlantSpecies} <: Agent{P}
    const id::Int
    size::Int
    const max_size::Int
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


########## Eating / Dying / Reproducing  ########################################

function eat!(wolf::Animal{Wolf}, sheep::Animal{Sheep}, w::World)
    wolf.energy += sheep.energy * wolf.Δenergy
    kill_agent!(sheep,w)
end
function eat!(sheep::Animal{Sheep}, grass::Plant{Grass}, ::World)
    sheep.energy += grass.size * sheep.Δenergy
    grass.size = 0
end
eat!(::Animal, ::Nothing, ::World) = nothing

kill_agent!(a::Agent, w::World) = delete!(w.agents, a.id)

function find_agent(::Type{A}, w::World) where A<:Agent
    dict = get(w.agents, tosym(A), nothing)
    if !isnothing(dict)
        as = dict |> values |> collect
        isempty(as) ? nothing : rand(as)
    else
        nothing
    end
end

find_agent(::Type{P}, w::World) where P<:PlantSpecies = find_agent(Plant{P}, w)
find_agent(::Type{A}, w::World) where A<:AnimalSpecies = find_agent(Animal{A}, w)

find_food(::Animal{Wolf}, w::World) = find_agent(Sheep, w)
find_food(::Animal{Sheep}, w::World) = find_agent(Grass, w)

function find_mate(a::A, w::World) where A<:Animal
    dict = get(w.agents, tosym(A), nothing)
    if !isnothing(dict)
        as = filter(x -> a.sex != x.sex, dict |> values |> collect)
        isempty(as) ? nothing : rand(as)
    else
        nothing
    end
end

function reproduce!(a::Animal{A}, w::World) where A
    m = find_mate(a,w)
    if !isnothing(m)
        a.energy = a.energy / 2
        new_id = w.max_id + 1
        ŝ = Animal{A}(new_id, a.energy, a.Δenergy, a.reprprob, a.foodprob, randsex())
        getfield(w.agents, tosym(ŝ))[ŝ.id] = ŝ
        w.max_id = new_id
        return ŝ
    else
        nothing
    end
end

##########  Stepping through time  #############################################

function agent_step!(p::Plant, ::World)
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
    if a.energy <= 0
        kill_agent!(a,w)
        return
    end
    if rand() <= a.reprprob
        reproduce!(a,w)
    end
    return a
end

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


##########  Counting agents  ####################################################

agent_count(p::Plant) = p.size / p.max_size
agent_count(::Animal) = 1
agent_count(as::Vector{<:Agent}) = sum(agent_count,as)

function agent_count(w::World)
    function op(d::Dict,a::A) where A<:Agent
        if A in keys(d)
            d[A] += agent_count(a)
        else
            d[A] = agent_count(a)
        end
        return d
    end
    foldl(op, w.agents |> values |> collect, init=Dict())
end

# for accessing NamedTuple in World
tosym(::T) where T<:Animal = tosym(T)

# NOTE: needed for type stability
# TODO: do this with meta programming
tosym(::Type{Animal{Wolf}}) = Symbol("Wolf")
tosym(::Type{Animal{Sheep}}) = Symbol("Sheep")
tosym(::Type{Plant{Grass}}) = Symbol("Grass")
