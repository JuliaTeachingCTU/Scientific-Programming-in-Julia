using BenchmarkTools

include("Ecosystem.jl")

function make_counter()
    n = 0
    counter() = n += 1
end

function create_world()
    n_grass  = 1_000
    n_sheep  = 40
    n_wolves = 4

    nextid = make_counter()

    World(vcat(
        [Grass(nextid()) for _ in 1:n_grass],
        [Sheep(nextid()) for _ in 1:n_sheep],
        [Wolf(nextid()) for _ in 1:n_wolves],
    ))
end
world = create_world();

world_step!(world)
@profview for i=1:100 world_step!(world) end
