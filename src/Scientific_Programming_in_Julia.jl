module Scientific_Programming_in_Julia

include("Ecosystem.jl")

using .Ecosystem
export Grass, Sheep, Wolf, World
export agent_step!, agent_count

end
