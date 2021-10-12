module Ecosystem

using StatsBase
using EcosystemCore

export Grass, Sheep, Wolf, World, Mushroom
export agent_step!, agent_count, world_step!, simulate!, every_nth


getargs(S::Type{<:Species},args::Tuple) = args
getargs(S::Type{<:AnimalSpecies},a::NamedTuple) = (a.energy,a.Δenergy,a.reprprob,a.foodprob)
getargs(S::Type{<:PlantSpecies},a::NamedTuple) = (a.max_size,)

function EcosystemCore.World(configs::Tuple...)
    ids_stop  = cumsum([n for (_,n,_) in configs])
    ids_start = vcat([1], ids_stop[1:end-1] .+ 1)
    xs = map(zip(ids_start, ids_stop, configs)) do (start,stop,config)
        (S,_,args) = config
        [S(id,getargs(S,args)...) for id in start:stop]
    end
    World(reduce(vcat, xs))
end


abstract type Mushroom <: PlantSpecies end
Base.show(io::IO,::Type{Mushroom}) = print(io,"🍄")

EcosystemCore.eats(::Animal{Sheep},::Plant{Mushroom}) = true
function EcosystemCore.eat!(s::Animal{Sheep}, m::Plant{Mushroom}, w::World)
    incr_energy!(s, -size(m)*Δenergy(s))
    m.size = 0
end


function simulate!(world::World, iters::Int; callbacks=[])
    for i in 1:iters
        world_step!(world)
        for cb in callbacks
            cb(world)
        end
    end
end


countsym(g::T) where T<:Plant = fully_grown(g) ? nameof(T) : :NoCount
countsym(::T) where T<:Animal = nameof(T)

# function agent_count(as::Vector{Agent})
#     cs = StatsBase.countmap(map(countsym,as))
#     delete!(cs,:NoCount)
# end

agent_count(p::Plant) = size(p)/EcosystemCore.max_size(p)
agent_count(::Animal) = 1
agent_count(as::Vector{<:Agent}) = sum(agent_count,as)

function agent_count(w::World)
    function op(d::Dict,a::Agent{S}) where S<:Species
        n = nameof(S)
        if n in keys(d)
            d[n] += agent_count(a)
        else
            d[n] = agent_count(a)
        end
        return d
    end
    foldl(op, w.agents |> values |> collect, init=Dict{Symbol,Real}())
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
