module Scientific_Programming_in_Julia

include("Ecosystem.jl")

using .Ecosystem
export Grass, Sheep, Wolf, World, PoisonedGrass, GenderedSheep
export agent_step!, agent_count, simulate!, every_nth

end
