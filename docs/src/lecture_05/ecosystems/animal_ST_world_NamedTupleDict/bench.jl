using BenchmarkTools
using Random
Random.seed!(0)

include("Ecosystem.jl")

sheep = Sheep(1,1,1,1,1,Female)
sheep2 = Sheep(3001,1,1,1,1,Male)
world = World(vcat([sheep,sheep2], [Grass(i) for i=2:3000]))

# check that something is returned
@info "check returns" find_food(sheep, world) reproduce!(sheep, world)

# check type stability
@code_warntype find_food(sheep, world)
@code_warntype reproduce!(sheep, world)

# benchmark
sheep = Sheep(1,1,1,1,1,Female)
sheep2 = Sheep(3001,1,1,1,1,Male)
world = World(vcat([sheep,sheep2], [Grass(i) for i=2:3000]))
@btime find_food($sheep, $world)

sheep = Sheep(1,1,1,1,1,Female)
sheep2 = Sheep(3001,1,1,1,1,Male)
world = World(vcat([sheep,sheep2], [Grass(i) for i=2:3000]))
@btime reproduce!($sheep, $world)
