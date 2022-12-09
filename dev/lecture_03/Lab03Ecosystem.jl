using StatsBase

abstract type Species end
abstract type Agent{S<:Species} end

abstract type PlantSpecies <: Species end
abstract type Grass <: PlantSpecies end

abstract type AnimalSpecies <: Species end
abstract type Sheep <: AnimalSpecies end
abstract type Wolf <: AnimalSpecies end

abstract type Sex end
abstract type Male <: Sex end
abstract type Female <: Sex end

id(a::Agent) = a.id


##########  World Definition  ##################################################

mutable struct World{A<:Agent}
    agents::Dict{Int,A}
    max_id::Int
end

function World(agents::Vector{<:Agent})
    ids = id.(agents)
    length(unique(ids)) == length(agents) || error("Not all agents have unique IDs!")
    World(Dict(id(a)=>a for a in agents), maximum(ids))
end

function world_step!(world::World)
    for id in deepcopy(keys(world.agents))
        !haskey(world.agents,id) && continue
        a = world.agents[id]
        agent_step!(a,world)
    end
end

function Base.show(io::IO, w::World)
    println(io, typeof(w))
    for (_,a) in w.agents
        println(io,"  $a")
    end
end


##########  Plant Definition  ##################################################

mutable struct Plant{P<:PlantSpecies} <: Agent{P}
    id::Int
    size::Int
    max_size::Int
end

Base.size(a::Plant) = a.size
max_size(a::Plant) = a.max_size
grow!(a::Plant) = a.size += 1

# constructor for all Plant{<:PlantSpecies} callable as PlantSpecies(...)
(A::Type{<:PlantSpecies})(id, s, m) = Plant{A}(id,s,m)
(A::Type{<:PlantSpecies})(id, m) = (A::Type{<:PlantSpecies})(id,rand(1:m),m)

function agent_step!(a::Plant, w::World)
    if size(a) != max_size(a)
        grow!(a)
    end
end

function Base.show(io::IO, p::Plant{P}) where P
    x = size(p)/max_size(p) * 100
    print(io,"$P  #$(id(p)) $(round(Int,x))% grown")
end

Base.show(io::IO, ::Type{Grass}) = print(io,"🌿")


##########  Animal Definition  #################################################

mutable struct Animal{A<:AnimalSpecies,S<:Sex} <: Agent{A}
    id::Int
    energy::Float64
    Δenergy::Float64
    reprprob::Float64
    foodprob::Float64
end

energy(a::Animal) = a.energy
Δenergy(a::Animal) = a.Δenergy
reprprob(a::Animal) = a.reprprob
foodprob(a::Animal) = a.foodprob
energy!(a::Animal, e) = a.energy = e
incr_energy!(a::Animal, Δe) = energy!(a, energy(a)+Δe)

function (A::Type{<:AnimalSpecies})(id::Int, E, ΔE, pr, pf, S=rand(Bool) ? Female : Male)
    Animal{A,S}(id,E,ΔE,pr,pf)
end

function agent_step!(a::Animal, w::World)
    incr_energy!(a,-1)
    if rand() <= foodprob(a)
        dinner = find_food(a,w)
        eat!(a, dinner, w)
    end
    if energy(a) <= 0
        kill_agent!(a,w)
        return
    end
    if rand() <= reprprob(a)
        reproduce!(a,w)
    end
    return a
end

function find_rand(f, w::World)
    xs = filter(f, w.agents |> values |> collect)
    isempty(xs) ? nothing : sample(xs)
end

find_food(a::Animal, w::World) = find_rand(x->eats(a,x),w)

eats(::Animal{Sheep},p::Plant{Grass}) = size(p)>0
eats(::Animal{Wolf},::Animal{Sheep}) = true
eats(::Agent,::Agent) = false

function eat!(a::Animal{Wolf}, b::Animal{Sheep}, w::World)
    incr_energy!(a, energy(b)*Δenergy(a))
    kill_agent!(b,w)
end
function eat!(a::Animal{Sheep}, b::Plant{Grass}, w::World)
    incr_energy!(a, size(b)*Δenergy(a))
    b.size = 0
end
eat!(::Animal,::Nothing,::World) = nothing

function reproduce!(a::A, w::World) where A<:Animal
    b = find_mate(a,w)
    if !isnothing(b)
        energy!(a, energy(a)/2)
        a_vals = [getproperty(a,n) for n in fieldnames(A) if n!=:id]
        new_id = w.max_id + 1
        â = A(new_id, a_vals...)
        w.agents[id(â)] = â
        w.max_id = new_id
    end
end

find_mate(a::Animal, w::World) = find_rand(x->mates(a,x),w)

function mates(a,b)
    error("""You have to specify the mating behaviour of your agents by overloading `mates` e.g. like this:

        mates(a::Animal{S,Female}, b::Animal{S,Male}) where S<:Species = true
        mates(a::Animal{S,Male}, b::Animal{S,Female}) where S<:Species = true
        mates(a::Agent, b::Agent) = false
    """)
end

kill_agent!(a::Animal, w::World) = delete!(w.agents, id(a))

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
