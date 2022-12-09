# using Ecosystem
using Scientific_Programming_in_Julia.Ecosystem

function create_world()
    n_grass       = 500
    regrowth_time = 17.0

    n_sheep         = 100
    Δenergy_sheep   = 5.0
    sheep_reproduce = 0.5
    sheep_foodprob  = 0.4

    n_wolves       = 8
    Δenergy_wolf   = 17.0
    wolf_reproduce = 0.03
    wolf_foodprob  = 0.02

    gs = [Grass(id, regrowth_time) for id in 1:n_grass];
    ss = [Sheep(id, 2*Δenergy_sheep, Δenergy_sheep, sheep_reproduce, sheep_foodprob) for id in n_grass+1:n_grass+n_sheep];
    ws = [Wolf(id, 2*Δenergy_wolf, Δenergy_wolf, wolf_reproduce, wolf_foodprob) for id in n_grass+n_sheep+1:n_grass+n_sheep+n_wolves];
    World(vcat(gs, ss, ws))
end
world = create_world();
simulate!(world, 10)