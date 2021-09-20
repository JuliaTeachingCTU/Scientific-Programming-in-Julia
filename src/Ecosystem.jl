module Ecosystem

using StatsBase
using EcosystemCore

export Grass, Sheep, Wolf, World, PoisonedGrass, ⚥Sheep
export agent_step!, agent_count, simulate!, every_nth

mutable struct PoisonedGrass <: AbstractPlant
    fully_grown::Bool
    regrowth_time::Int
    countdown::Int
end
PoisonedGrass(t) = PoisonedGrass(false, t, rand(1:t))
PoisonedGrass() = PoisonedGrass(2)

EcosystemCore.eats(::Sheep,::PoisonedGrass) = true

function EcosystemCore.eat!(sheep::Sheep, grass::PoisonedGrass, w::World)
     if fully_grown(grass)
        fully_grown!(grass, false)
        incr_energy!(sheep, -Δenergy(sheep))
    end
end


struct ⚥Sheep{T<:Real} <: AbstractAnimal
    sheep::Sheep{T}
    gender::Symbol
end
⚥Sheep(E,ΔE,pr,pf,gender) = ⚥Sheep(Sheep(E,ΔE,pr,pf),gender)
⚥Sheep(E,ΔE,pr,pf) = ⚥Sheep(E,ΔE,pr,pf,sample([:male,:female]))

EcosystemCore.energy(g::⚥Sheep) = energy(g.sheep)
EcosystemCore.energy!(g::⚥Sheep, e) = energy!(g.sheep, e)
EcosystemCore.Δenergy(g::⚥Sheep, e) = Δenergy(g.sheep, e)
EcosystemCore.incr_energy!(g::⚥Sheep, Δe) = incr_energy!(g.sheep, Δe)
EcosystemCore.reproduction_prob(g::⚥Sheep) = reproduction_prob(g.sheep)
EcosystemCore.food_prob(g::⚥Sheep) = food_prob(g.sheep)

EcosystemCore.eats(::⚥Sheep, ::Grass) = true
EcosystemCore.eats(::⚥Sheep, ::PoisonedGrass) = true
EcosystemCore.eat!(s::⚥Sheep, g::AbstractPlant, w::World) = eat!(s.sheep, g, w)

mates(a::AbstractPlant, ::⚥Sheep) = false
mates(a::AbstractAnimal, ::⚥Sheep) = false
mates(g1::⚥Sheep, g2::⚥Sheep) = g1.gender != g2.gender
function find_mate(g::⚥Sheep, w::World)
    ms = filter(a->mates(a,g), w.agents)
    isempty(ms) ? nothing : sample(ms)
end

function EcosystemCore.reproduce!(s::⚥Sheep, w::World)
    m = find_mate(s,w)
    if !isnothing(m)
        energy!(s, energy(s)/2)
        # TODO: should probably mix s/m
        push!(w.agents, deepcopy(s))
    end
end



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

end
