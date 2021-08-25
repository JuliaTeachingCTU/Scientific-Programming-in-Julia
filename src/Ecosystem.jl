module Ecosystem

using StatsBase

export Grass, Sheep, Wolf, World
export agent_step!

abstract type AbstractAgent end
abstract type AbstractPlant <: AbstractAgent end
abstract type AbstractAnimal <: AbstractAgent end

mutable struct Grass <: AbstractPlant
    fully_grown::Bool
    regrowth_time::Int
    countdown::Int
end
Grass(t) = Grass(false, t, rand(1:t))
Grass() = Grass(2)

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

function agent_step!(a::Grass, w::World)
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
    a.energy -= 1
    dinner = find_food(a,w)
    eat!(a, dinner, w)
    if a.energy < 0
        kill_agent!(a,w)
        return
    end
    if rand() <= a.reproduction_prob
        reproduce!(a,w)
    end
    return a
end

function find_food(a::T, w::World) where T<:AbstractAnimal
    if rand() <= a.food_prob
        as = filter(x->isa(x,eats(T)), w.agents)
        isempty(as) ? nothing : sample(as)
    end
end

eats(::Type{<:Sheep}) = Grass
eats(::Type{<:Wolf}) = Sheep

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
eat!(::AbstractAnimal,::Nothing,::World) = nothing

function reproduce!(a::AbstractAnimal, w::World)
    a.energy /= 2
    push!(w.agents, deepcopy(a))
end

kill_agent!(a::AbstractAnimal, w::World) = deleteat!(w.agents, findall(x->x==a, w.agents))

count(g::Ecosystem.AbstractPlant) = g.fully_grown ? 1 : 0
count(::Ecosystem.AbstractAnimal) = 1

count(as::Vector{<:Ecosystem.AbstractAgent}) = map(count,as) |> sum

function count(as::Vector{Ecosystem.AbstractAgent})
    Ts = unique(typeof.(as))
    cs = map(Ts) do T
        _as = Vector{T}(filter(x->isa(x,T), as))
        T => count(_as)
    end
    Dict(cs...)
end

end
