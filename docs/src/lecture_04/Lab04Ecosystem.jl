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

function find_mate(a::Animal, w::World)
    ms = filter(x->mates(x,a), w.agents |> values |> collect)
    isempty(ms) ? nothing : sample(ms)
end
mates(a::Animal{A}, b::Animal{A}) where A<:AnimalSpecies = a.sex != b.sex
mates(::Agent, ::Agent) = false

function reproduce!(a::Animal{A}, w::World) where A
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

# finding food / who eats who
function find_food(a::Animal, w::World)
    as = filter(x -> eats(a,x), w.agents |> values |> collect)
    isempty(as) ? nothing : sample(as)
end
eats(::Animal{Sheep},g::Plant{Grass}) = g.size > 0
eats(::Animal{Wolf},::Animal{Sheep}) = true
eats(::Agent,::Agent) = false


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
    function op(d::Dict,a::Agent{S}) where S<:Species
        n = nameof(S)
        d[n] = haskey(d,n) ? d[n]+agent_count(a) : agent_count(a)
        return d
    end
    reduce(op, w.agents |> values, init=Dict{Symbol,Float64}())
end
