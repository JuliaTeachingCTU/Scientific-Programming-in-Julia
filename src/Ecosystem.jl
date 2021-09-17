module Ecosystem

using StatsBase

export Grass, Sheep, Wolf, World, PoisonedGrass, ⚥Sheep
export agent_step!, agent_count, simulate!, every_nth

abstract type AbstractAgent end
abstract type AbstractPlant <: AbstractAgent end
abstract type AbstractAnimal <: AbstractAgent end

fully_grown(a::AbstractPlant) = a.fully_grown
fully_grown!(a::AbstractPlant, b::Bool) = a.fully_grown = b
countdown(a::AbstractPlant) = a.countdown
countdown!(a::AbstractPlant, c::Int) = a.countdown = c
incr_countdown!(a::AbstractPlant, Δc::Int) = countdown!(a, countdown(a)+Δc)
reset!(a::AbstractPlant) = a.countdown = a.regrowth_time

energy(a::AbstractAnimal) = a.energy
energy!(a::AbstractAnimal, e) = a.energy = e
incr_energy!(a::AbstractAnimal, Δe) = energy!(a, energy(a)+Δe)
Δenergy(a::AbstractAnimal) = a.Δenergy
reproduction_prob(a::AbstractAnimal) = a.reproduction_prob
food_prob(a::AbstractAnimal) = a.food_prob

mutable struct Grass <: AbstractPlant
    fully_grown::Bool
    regrowth_time::Int
    countdown::Int
end
Grass(t) = Grass(false, t, rand(1:t))
Grass() = Grass(2)

mutable struct PoisonedGrass <: AbstractPlant
    fully_grown::Bool
    regrowth_time::Int
    countdown::Int
end
PoisonedGrass(t) = PoisonedGrass(false, t, rand(1:t))
PoisonedGrass() = PoisonedGrass(2)


mutable struct Sheep{T<:Real} <: AbstractAnimal
    energy::T
    Δenergy::T
    reproduction_prob::T
    food_prob::T
end

mutable struct Wolf{T<:Real} <: AbstractAnimal
    energy::T
    Δenergy::T
    reproduction_prob::T
    food_prob::T
end

struct World{T<:AbstractAgent}
    agents::Vector{T}
end
function Base.show(io::IO, w::World)
    println(io, typeof(w))
    map(a->println(io,"  $a"),w.agents)
end

function agent_step!(a::AbstractPlant, w::World)
    if !fully_grown(a)
        if countdown(a) <= 0
            fully_grown!(a,true)
            reset!(a)
        else
            incr_countdown!(a,-1)
        end
    end
    return a
end

function agent_step!(a::AbstractAnimal, w::World)
    incr_energy!(a,-1)
    dinner = find_food(a,w)
    eat!(a, dinner, w)
    if energy(a) < 0
        kill_agent!(a,w)
        return
    end
    if rand() <= reproduction_prob(a)
        reproduce!(a,w)
    end
    return a
end

function find_food(a::AbstractAnimal, w::World)
    if rand() <= food_prob(a)
        as = filter(x->eats(a,x), w.agents)
        isempty(as) ? nothing : sample(as)
    end
end

eats(::Sheep,::Grass) = true
eats(::Sheep,::PoisonedGrass) = true
eats(::Wolf,::Sheep) = true
eats(::AbstractAgent,::AbstractAgent) = false

function eat!(wolf::Wolf, sheep::Sheep, w::World)
    kill_agent!(sheep,w)
    incr_energy!(wolf, Δenergy(wolf))
end
function eat!(sheep::Sheep, grass::Grass, w::World)
    if fully_grown(grass)
        fully_grown!(grass, false)
        incr_energy!(sheep, Δenergy(sheep))
    end
end
function eat!(sheep::Sheep, grass::PoisonedGrass, w::World)
     if fully_grown(grass)
        fully_grown!(grass, false)
        incr_energy!(sheep, -Δenergy(sheep))
    end
end
eat!(::AbstractAnimal,::Nothing,::World) = nothing

function reproduce!(a::AbstractAnimal, w::World)
    energy!(a, energy(a)/2)
    push!(w.agents, deepcopy(a))
end

kill_agent!(a::AbstractAnimal, w::World) = deleteat!(w.agents, findall(x->x==a, w.agents))

countsym(g::T) where T<:AbstractPlant = fully_grown(g) ? nameof(T) : :NoCount
countsym(::T) where T<:AbstractAnimal = nameof(T)

# function agent_count(as::Vector{AbstractAgent})
#     cs = StatsBase.countmap(map(countsym,as))
#     delete!(cs,:NoCount)
# end

agent_count(g::AbstractPlant) = fully_grown(g) ? 1 : 0
agent_count(::AbstractAnimal) = 1
agent_count(as::Vector{<:AbstractAgent}) = sum(agent_count,as)

function agent_count(w::World)
    function op(d::Dict,a::T) where T<:AbstractAgent
        n = nameof(T)
        if n in keys(d)
            d[n] += agent_count(a)
        else
            d[n] = agent_count(a)
        end
        return d
    end
    foldl(op, w.agents, init=Dict{Symbol,Int}())
end

function simulate!(w::World, iters::Int; callbacks=[])
    for i in 1:iters
        for a in w.agents
            agent_step!(a,w)
        end
        for cb in callbacks
            cb(w)
        end
    end
end

function every_nth(f::Function, n::Int)
    i = 1
    function callback(w::World)
        # display(i) # comment this out to see out the counter increases
        if i == n
            f(w)
            i = 1
        else
            i += 1
        end
    end
end


struct ⚥Sheep{T<:Real} <: AbstractAnimal
    sheep::Sheep{T}
    gender::Symbol
end
⚥Sheep(E,ΔE,pr,pf,gender) = ⚥Sheep(Sheep(E,ΔE,pr,pf),gender)
⚥Sheep(E,ΔE,pr,pf) = ⚥Sheep(E,ΔE,pr,pf,sample([:male,:female]))

energy(g::⚥Sheep) = energy(g.sheep)
energy!(g::⚥Sheep, e) = energy!(g.sheep, e)
Δenergy(g::⚥Sheep, e) = Δenergy(g.sheep, e)
incr_energy!(g::⚥Sheep, Δe) = incr_energy!(g.sheep, Δe)
reproduction_prob(g::⚥Sheep) = reproduction_prob(g.sheep)
food_prob(g::⚥Sheep) = food_prob(g.sheep)

eats(::⚥Sheep, ::Grass) = true
eats(::⚥Sheep, ::PoisonedGrass) = true
eat!(s::⚥Sheep, g::AbstractPlant, w::World) = eat!(s.sheep, g, w)

mates(a::AbstractPlant, ::⚥Sheep) = false
mates(a::AbstractAnimal, ::⚥Sheep) = false
mates(g1::⚥Sheep, g2::⚥Sheep) = g1.gender != g2.gender
function find_mate(g::⚥Sheep, w::World)
    ms = filter(a->mates(a,g), w.agents)
    isempty(ms) ? nothing : sample(ms)
end

function reproduce!(s::⚥Sheep, w::World)
    m = find_mate(s,w)
    if !isnothing(m)
        energy!(s, energy(s)/2)
        # TODO: should probably mix s/m
        push!(w.agents, deepcopy(s))
    end
end

end
