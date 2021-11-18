module Scientific_Programming_in_Julia

include("ReverseDiff.jl")
using .ReverseDiff
export track, accum!, σ
export TrackedArray, TrackedMatrix, TrackedVector

include("Ecosystem.jl")

using .Ecosystem
export Grass, Sheep, Wolf, World, Mushroom
export agent_step!, agent_count, world_step!, every_nth

end
