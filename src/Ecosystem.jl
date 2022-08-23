module Ecosystem

using StatsBase
using EcosystemCore

export Grass, Sheep, Wolf, World, Mushroom
export agent_step!, agent_count, world_step!, simulate!, every_nth


# abstract type Mushroom <: PlantSpecies end
# Base.show(io::IO,::Type{Mushroom}) = print(io,"🍄")
# 
# EcosystemCore.eats(::Animal{Sheep},::Plant{Mushroom}) = true
# function EcosystemCore.eat!(s::Animal{Sheep}, m::Plant{Mushroom}, w::World)
#     incr_energy!(s, -size(m)*Δenergy(s))
#     m.size = 0
# end

EcosystemCore.mates(::Animal{S,Female}, ::Animal{S,Male}) where S<:Species = true
EcosystemCore.mates(::Animal{S,Male}, ::Animal{S,Female}) where S<:Species = true
EcosystemCore.mates(a::Agent, b::Agent) = false

function simulate!(world::World, iters::Int; callbacks=[])
    for i in 1:iters
        world_step!(world)
        for cb in callbacks
            cb(world)
        end
    end
end

agent_count(p::Plant) = size(p)/EcosystemCore.max_size(p)
agent_count(a::Animal) = 1
function agent_count(as::AbstractVector{<:Agent})
    if length(as) > 0
        sum(agent_count,as) |> Float64
    else
        NaN
    end
end
speciessym(::Type{A}) where A<:Agent = EcosystemCore.tosym(A)
speciessym(::Dict{Int,A}) where A<:Agent = speciessym(A)
agent_count(w::World) = Dict(speciessym(as) => agent_count(as |> values |> collect) for as in w.agents)
    # function op(d::Dict,a::Agent{S}) where S<:Species
    #     n = nameof(S)
    #     if n in keys(d)
    #         d[n] += agent_count(a)
    #     else
    #         d[n] = agent_count(a)
    #     end
    #     return d
    # end
    # foldl(op, w.agents |> values |> collect, init=Dict{Symbol,Real}())
#end

function every_nth(f::Function, n::Int)
    i = 1
    function callback(args...)
        # display(i) # comment this out to see out the counter increases
        if i == n
            f(args...)
            i = 1
        else
            i += 1
        end
    end
end

end
