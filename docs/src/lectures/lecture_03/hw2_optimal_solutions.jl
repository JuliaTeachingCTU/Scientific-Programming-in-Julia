##  agent_count
### recommended option
agent_count(a::Animal) = 1.0
agent_count(p::Plant) = p.size / p.max_size

## agent_count(agents::Vector{<:Agent})
### recommended option
agent_count(as::Vector{<:Agent}) = sum(agent_count,as)

## agent_count(w::World)
### recommended option (little fancy)
function agent_count(w::World)
    function op(d::Dict,a::A) where A<:Agent
        n = nameof(A)
        d[n] = haskey(d,n) ? d[n]+agent_count(a) : agent_count(a)
        return d
    end
    reduce(op, w.agents |> values, init=Dict{Symbol,Float64}())
end
