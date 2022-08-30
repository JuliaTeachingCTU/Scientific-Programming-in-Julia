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


##########  Animals  ###########################################################

mutable struct Animal{A<:AnimalSpecies,S<:Sex} <: Agent{A}
    id::Int
    energy::Float64
    Δenergy::Float64
    reprprob::Float64
    foodprob::Float64
end

function (A::Type{<:AnimalSpecies})(id::Int,E::T,ΔE::T,pr::T,pf::T) where T
    S = rand(Bool) ? Female : Male
    Animal{A,S}(id,E,ΔE,pr,pf)
end

# get the per species defaults back
Sheep(id; E=4.0, ΔE=0.2, pr=0.8, pf=0.6) = Sheep(id, E, ΔE, pr, pf)
Wolf(id; E=10.0, ΔE=8.0, pr=0.1, pf=0.2) = Wolf(id, E, ΔE, pr, pf)


function Base.show(io::IO, a::Animal{A,S}) where {A<:AnimalSpecies,S<:Sex}
    e = a.energy
    d = a.Δenergy
    pr = a.reprprob
    pf = a.foodprob
    print(io, "$A$S #$(a.id) E=$e ΔE=$d pr=$pr pf=$pf")
end

# note that for new species/sexes we will only have to overload `show` on the
# abstract species/sex types like below!
Base.show(io::IO, ::Type{Sheep}) = print(io,"🐑")
Base.show(io::IO, ::Type{Wolf}) = print(io,"🐺")
Base.show(io::IO, ::Type{Male}) = print(io,"♂")
Base.show(io::IO, ::Type{Female}) = print(io,"♀")


##########  Plants  #############################################################

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

########## Eating / Dying / Reproducing  ########################################

function eat!(sheep::Animal{Sheep}, grass::Animal{Grass}, w::World)
    sheep.energy += grass.size * sheep.Δenergy
    grass.size = 0
end
function eat!(wolf::Animal{Wolf}, sheep::Animal{Sheep}, w::World)
    wolf.energy += sheep.energy * wolf.Δenergy
    kill_agent!(sheep,w)
end
eat!(::Animal, ::Nothing, ::World) = nothing

kill_agent!(a::Agent, w::World) = delete!(w.agents, a.id)

mates(a::Animal{A,Female}, b::Animal{A,Male}) where A<:AbstracSpecies = true
mates(a::Animal{A,Male}, b::Animal{A,Female}) where A<:AbstracSpecies = true
mates(::Agent, ::Agent) = false

function find_mate(a::Animal, w::World)
    ms = filter(x->mates(x,a), w.agents |> values |> collect)
    isempty(ms) ? nothing : sample(ms)
end

function reproduce!(a::Animal{A,S}, w::World) where {A,S}
    m = find_mate(a,w)
    if !isnothing(m)
        a.energy = a.energy / 2
        vals = [getproperty(a,n) for n in fieldnames(Animal) if n!=:id]
        new_id = w.max_id + 1
        ŝ = Animal{A,S}(new_id, vals...)
        w.agents[ŝ.id] = ŝ
        w.max_id = new_id
    end
end


##########  Counting agents  ####################################################

agent_count(p::Plant) = p.size / p.max_size
agent_count(::Animal) = 1
agent_count(as::Vector{<:Agent}) = sum(agent_count,as)

function agent_count(w::World)
    function op(d::Dict,a::A) where A<:Agent
        n = nameof(A)
        if n in keys(d)
            d[n] += agent_count(a)
        else
            d[n] = agent_count(a)
        end
        return d
    end
    foldl(op, w.agents |> values |> collect, init=Dict{Symbol,Real}())
end
