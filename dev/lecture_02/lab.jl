include("Lab02Ecosystem.jl")


w = World([Wolf(2), Grass(1), Grass(3)])
# World{Agent}
#   🐺♂ #2 E=10.0 ΔE=8.0 pr=0.1 pf=0.2
#   🌿  #3 60% grown
#   🌿  #1 100% grown


## a> agent_count(w)
# Dict{Symbol, Real} with 2 entries:
#   :Animal => 1
#   :Plant  => 1.6

agent_count(x::Animal) = 1
agent_count(x::Plant) = x.size / x.max_size

function agent_count(w::World)
    counts = Dict()
    for (_,a) in w.agents
        n = a |> typeof |> nameof
        haskey(counts, n) ? counts[n] += agent_count(a) : counts[n] = agent_count(a)
    end
    return counts
end
