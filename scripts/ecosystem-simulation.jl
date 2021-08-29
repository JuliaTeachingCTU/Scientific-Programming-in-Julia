using Scientific_Programming_in_Julia
using Random
Random.seed!(4)

n_sheep  = 100
n_wolves = 8
n_grass = 500
regrowth_time = 17.0
Δenergy_sheep = 5.0
Δenergy_wolf = 17.0
wolf_reproduce = 0.03
sheep_reproduce = 0.5
wolf_foodprob = 0.02
sheep_foodprob = 0.4

gs = [Grass(true,regrowth_time,regrowth_time) for _ in 1:n_grass]
pgs = [PoisonedGrass(true,regrowth_time,regrowth_time) for _ in 1:65]
ss = [Sheep(2*Δenergy_sheep,Δenergy_sheep,sheep_reproduce, sheep_foodprob) for _ in 1:n_sheep]
ws = [Wolf(2*Δenergy_wolf,Δenergy_wolf,wolf_reproduce, wolf_foodprob) for _ in 1:n_wolves]
as = vcat(gs,pgs,ss,ws)

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

simulate!(w, 200, callbacks=cbs)

using Plots
plt = plot()
for (n,c) in counts
    plot!(plt, c, label="$n", lw=2)
end
# savefig(plt, "docs/src/lecture_02/pred-prey.png")
# savefig(plt, "docs/src/lecture_03/pred-prey.png")
display(plt)
