module Ecosystem

using StatsBase

export Grass, Sheep, Wolf, World, PoisonedGrass, GenderedSheep
export agent_step!, agent_count, simulate!, every_nth

abstract type AbstractAgent end
abstract type AbstractPlant <: AbstractAgent end
abstract type AbstractAnimal <: AbstractAgent end

energy(a::AbstractAnimal) = a.energy
energy!(a::AbstractAnimal, ΔE) = a.energy += ΔE
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
    if !a.fully_grown
        if a.countdown <= 0
            a.fully_grown = true
            a.countdown = a.regrowth_time
        else
            a.countdown -= 1
        end
    end
    return a
end

function agent_step!(a::A, w::World) where A<:AbstractAnimal
    energy!(a, -1)
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

function find_food(a::T, w::World) where T<:AbstractAnimal
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
    wolf.energy += wolf.Δenergy
end
function eat!(sheep::Sheep, grass::Grass, w::World)
    if grass.fully_grown
        grass.fully_grown = false
        sheep.energy += sheep.Δenergy
    end
end
function eat!(sheep::Sheep, grass::PoisonedGrass, w::World)
     if grass.fully_grown
        grass.fully_grown = false
        sheep.energy -= sheep.Δenergy
    end
end
eat!(::AbstractAnimal,::Nothing,::World) = nothing

function reproduce!(a::AbstractAnimal, w::World)
    a.energy /= 2
    push!(w.agents, deepcopy(a))
end

kill_agent!(a::AbstractAnimal, w::World) = deleteat!(w.agents, findall(x->x==a, w.agents))

countsym(g::T) where T<:AbstractPlant = g.fully_grown ? nameof(T) : :NoCount
countsym(::T) where T<:AbstractAnimal = nameof(T)

# function agent_count(as::Vector{AbstractAgent})
#     cs = StatsBase.countmap(map(countsym,as))
#     delete!(cs,:NoCount)
# end

agent_count(g::AbstractPlant) = g.fully_grown ? 1 : 0
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


struct GenderedSheep{T<:Real} <: AbstractAnimal
    sheep::Sheep{T}
    gender::Symbol
end
GenderedSheep(E,ΔE,pr,pf,gender) = GenderedSheep(Sheep(E,ΔE,pr,pf),sample([:male,:female]))
GenderedSheep(E,ΔE,pr,pf) = GenderedSheep(E,ΔE,pr,pf,sample([:male,:female]))

energy(g::GenderedSheep) = energy(g.sheep)
energy!(g::GenderedSheep, ΔE) = energy!(g.sheep, ΔE)
reproduction_prob(g::GenderedSheep) = reproduction_prob(g.sheep)
food_prob(g::GenderedSheep) = food_prob(g.sheep)

eats(::GenderedSheep, ::Grass) = true
eats(::GenderedSheep, ::PoisonedGrass) = true
eat!(s::GenderedSheep, g::AbstractPlant, w::World) = eat!(s.sheep, g, w)

mates(a::AbstractPlant, ::GenderedSheep) = false
mates(a::AbstractAnimal, ::GenderedSheep) = false
mates(g1::GenderedSheep, g2::GenderedSheep) = g1.gender != g2.gender
function find_mate(g::GenderedSheep, w::World)
    ms = filter(a->mates(a,g), w.agents)
    isempty(ms) ? nothing : sample(ms)
end

function reproduce!(s::GenderedSheep, w::World)
    m = find_mate(s,w)
    if !isnothing(m)
        s.sheep.energy /= 2
        # TODO: should probably mix s/m
        push!(w.agents, deepcopy(s))
    end
end

end
