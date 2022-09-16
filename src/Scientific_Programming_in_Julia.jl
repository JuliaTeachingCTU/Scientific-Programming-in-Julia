module Scientific_Programming_in_Julia

include("ReverseDiff.jl")
using .ReverseDiff
export track, accum!, σ
export TrackedArray, TrackedMatrix, TrackedVector

end
