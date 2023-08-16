using BenchmarkTools
using Random
Random.seed!(0)

include("Ecosystem.jl")

sheep = Sheep(1,1,1,1,1,female)
sheep2 = Sheep(3001,1,1,1,1,male)
world = World(vcat([sheep,sheep2], [Grass(i) for i=2:3000]))

# check that something is returned
@info "check returns" find_food(sheep, world) reproduce!(sheep, world)

# check type stability
@code_warntype find_food(sheep, world)

sheep = Sheep(1,1,1,1,1,female)
sheep2 = Sheep(3001,1,1,1,1,male)
world = World(vcat([sheep,sheep2], [Grass(i) for i=2:3000]))
@code_warntype reproduce!(sheep, world)

# benchmark
sheep = Sheep(1,1,1,1,1,female)
sheep2 = Sheep(3001,1,1,1,1,male)
world = World(vcat([sheep,sheep2], [Grass(i) for i=2:3000]))
@btime find_food($sheep, $world)

sheep = Sheep(1,1,1,1,1,female)
sheep2 = Sheep(3001,1,1,1,1,male)
world = World(vcat([sheep,sheep2], [Grass(i) for i=2:3000]))
@btime reproduce!($sheep, $world)
