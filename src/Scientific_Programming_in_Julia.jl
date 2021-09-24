module Scientific_Programming_in_Julia

include("Ecosystem.jl")

using .Ecosystem
export Grass, Sheep, Wolf, World, Mushroom
export agent_step!, agent_count, world_step!

end
