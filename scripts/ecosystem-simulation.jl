using Scientific_Programming_in_Julia
using Random
Random.seed!(4)

n_grass       = 1000
regrowth_time = 21.0

n_sheep         = 10
Δenergy_sheep   = 4.0
sheep_reproduce = 0.3
sheep_foodprob  = 0.6

n_wolves       = 2
Δenergy_wolf   = 9.0
wolf_reproduce = 0.03
wolf_foodprob  = 0.12

gs = [Grass(id,true,regrowth_time,regrowth_time) for id in 1:n_grass]
ss = [Sheep(id,2*Δenergy_sheep,Δenergy_sheep,sheep_reproduce, sheep_foodprob) for id in (n_grass+1):(n_grass+n_sheep)]
ws = [Wolf(id,2*Δenergy_wolf,Δenergy_wolf,wolf_reproduce, wolf_foodprob) for id in (n_grass+n_sheep+1):(n_grass+n_sheep+n_wolves)]
#pgs = [PoisonedGrass(true,regrowth_time,regrowth_time) for _ in (n_grass+1):(n_grass+1)]
#ss = [Sheep(2*Δenergy_sheep,Δenergy_sheep,sheep_reproduce, sheep_foodprob) for _ in 1:n_sheep]
as = vcat(gs,ss,ws)

w = World(as)

counts = Dict(n=>[c] for (n,c) in agent_count(w))
function _save(w::World)
    for (n,c) in agent_count(w)
        push!(counts[n],c)
    end
end

logcb = every_nth(w->(@info agent_count(w)), 5)
savecb = every_nth(_save, 1)
cbs = [logcb, savecb]

simulate!(w, 500, callbacks=cbs)

using Plots
plt = plot()
for (n,c) in counts
    plot!(plt, c, label="$n", lw=2)
end
savefig(plt, "docs/src/lecture_02/pred-prey.png")
# savefig(plt, "docs/src/lecture_03/pred-prey.png")
display(plt)
