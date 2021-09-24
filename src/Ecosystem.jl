module Ecosystem

using StatsBase
using EcosystemCore

export Grass, Sheep, Wolf, World, PoisonedGrass, ⚥Sheep
export agent_step!, agent_count, simulate!, every_nth

mutable struct PoisonedGrass <: Plant
    id::Int
    fully_grown::Bool
    regrowth_time::Int
    countdown::Int
end
PoisonedGrass(id,t) = PoisonedGrass(id, false, t, rand(1:t))

EcosystemCore.eats(::Sheep,::PoisonedGrass) = true

function EcosystemCore.eat!(sheep::Sheep, grass::PoisonedGrass, w::World)
     if fully_grown(grass)
        fully_grown!(grass, false)
        incr_energy!(sheep, -Δenergy(sheep))
    end
end


struct ⚥Sheep <: Animal
    sheep::Sheep
    sex::Symbol
end
⚥Sheep(id,E,ΔE,pr,pf,sex) = ⚥Sheep(Sheep(id,E,ΔE,pr,pf),sex)

EcosystemCore.id(g::⚥Sheep) = EcosystemCore.id(g.sheep)
EcosystemCore.energy(g::⚥Sheep) = energy(g.sheep)
EcosystemCore.energy!(g::⚥Sheep, ΔE) = energy!(g.sheep, ΔE)
EcosystemCore.reproduction_prob(g::⚥Sheep) = reproduction_prob(g.sheep)
EcosystemCore.food_prob(g::⚥Sheep) = food_prob(g.sheep)

EcosystemCore.eats(::⚥Sheep, ::Grass) = true
# eats(::⚥Sheep, ::PoisonedGrass) = true
EcosystemCore.eat!(s::⚥Sheep, g::Plant, w::World) = eat!(s.sheep, g, w)

mates(a::Plant, ::⚥Sheep) = false
mates(a::Animal, ::⚥Sheep) = false
mates(g1::⚥Sheep, g2::⚥Sheep) = g1.sex != g2.sex
function find_mate(g::⚥Sheep, w::World)
    ms = filter(a->mates(a,g), w.agents |> values |> collect)
    isempty(ms) ? nothing : sample(ms)
end

function EcosystemCore.reproduce!(s::⚥Sheep, w::World)
    m = find_mate(s,w)
    if !isnothing(m)
        energy!(s, energy(s)/2)
        vals = [getproperty(s.sheep,n) for n in fieldnames(Sheep) if n!=:id]
        new_id = w.max_id + 1
        ŝ = ⚥Sheep(new_id, vals..., rand(Bool) ? :female : :male)
        w.agents[EcosystemCore.id(ŝ)] = ŝ
        w.max_id = new_id
    end
end




countsym(g::T) where T<:Plant = fully_grown(g) ? nameof(T) : :NoCount
countsym(::T) where T<:Animal = nameof(T)

# function agent_count(as::Vector{Agent})
#     cs = StatsBase.countmap(map(countsym,as))
#     delete!(cs,:NoCount)
# end

agent_count(g::Plant) = fully_grown(g) ? 1 : 0
agent_count(::Animal) = 1
agent_count(as::Vector{<:Agent}) = sum(agent_count,as)

function agent_count(w::World)
    function op(d::Dict,a::T) where T<:Agent
        n = nameof(T)
        if n in keys(d)
            d[n] += agent_count(a)
        else
            d[n] = agent_count(a)
        end
        return d
    end
    foldl(op, w.agents |> values |> collect, init=Dict{Symbol,Int}())
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
