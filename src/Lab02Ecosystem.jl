using StatsBase

abstract type Agent end
abstract type Animal <: Agent end
abstract type Plant <: Agent end

mutable struct World{A<:Agent}
    agents::Dict{Int,A}
    max_id::Int
end
function World(agents::Vector{<:Agent})
    World(Dict(id(a)=>a for a in agents), maximum(id.(agents)))
end

# optional: you can overload the `show` method to get custom
# printing of your World
function Base.show(io::IO, w::World)
    println(io, typeof(w))
    for (_,a) in w.agents
        println(io,"  $a")
    end
end

function world_step!(world::World)
    # make sure that we only iterate over IDs that already exist in the 
    # current timestep this lets us safely add agents
    ids = deepcopy(keys(world.agents))

    for id in ids
        # agents can be killed by other agents, so make sure that we are
        # not stepping dead agents forward
        !haskey(world.agents,id) && continue

        a = world.agents[id]
        agent_step!(a,world)
    end
end

function agent_step!(a::Plant, w::World)
    if size(a) != max_size(a)
        grow!(a)
    end
end

function agent_step!(a::Animal, w::World)
    incr_energy!(a,-1)
    if rand() <= food_prob(a)
        dinner = find_food(a,w)
        eat!(a, dinner, w)
    end
    if energy(a) <= 0
        kill_agent!(a,w)
        return
    end
    if rand() <= reproduction_prob(a)
        reproduce!(a,w)
    end
    return a
end

mutable struct Grass <: Plant
    id::Int
    size::Int
    max_size::Int
end

mutable struct Sheep <: Animal
    id::Int
    energy::Float64
    Δenergy::Float64
    reproduction_prob::Float64
    food_prob::Float64
end

mutable struct Wolf <: Animal
    id::Int
    energy::Float64
    Δenergy::Float64
    reproduction_prob::Float64
    food_prob::Float64
end

id(a::Agent) = a.id  # every agent has an ID so we can just define id for Agent here

Base.size(a::Plant) = a.size
max_size(a::Plant) = a.max_size
grow!(a::Plant) = a.size += 1

# get field values
energy(a::Animal) = a.energy
Δenergy(a::Animal) = a.Δenergy
reproduction_prob(a::Animal) = a.reproduction_prob
food_prob(a::Animal) = a.food_prob

# set field values
energy!(a::Animal, e) = a.energy = e
incr_energy!(a::Animal, Δe) = energy!(a, energy(a)+Δe)

function eat!(a::Sheep, b::Grass, w::World)
    incr_energy!(a, size(b)*Δenergy(a))
    kill_agent!(b,w)
end
function eat!(wolf::Wolf, sheep::Sheep, w::World)
    incr_energy!(wolf, energy(sheep)*Δenergy(wolf))
    kill_agent!(sheep,w)
end
eat!(a::Animal,b::Nothing,w::World) = nothing

kill_agent!(a::Plant, w::World) = a.size = 0
kill_agent!(a::Animal, w::World) = delete!(w.agents, id(a))

function find_food(a::Animal, w::World)
    as = filter(x->eats(a,x), w.agents |> values |> collect)
    isempty(as) ? nothing : sample(as)
end

eats(::Sheep,::Grass) = true
eats(::Wolf,::Sheep) = true
eats(::Agent,::Agent) = false

function reproduce!(a::A, w::World) where A<:Animal
    energy!(a, energy(a)/2)
    a_vals = [getproperty(a,n) for n in fieldnames(A) if n!=:id]
    new_id = w.max_id + 1
    â = A(new_id, a_vals...)
    w.agents[id(â)] = â
    w.max_id = new_id
end
