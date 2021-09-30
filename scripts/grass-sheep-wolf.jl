using Scientific_Programming_in_Julia
using EcosystemCore
using Random
Random.seed!(2)

n_grass  = 200
max_size = 10

n_sheep         = 10
Δenergy_sheep   = 0.2
energy_sheep    = 4.0
sheep_reproduce = 0.8
sheep_foodprob  = 0.6

n_wolves       = 2
Δenergy_wolf   = 8.0
energy_wolf    = 10.0
wolf_reproduce = 0.1
wolf_foodprob  = 0.2

# all animals can mate - regardless of sex
EcosystemCore.mates(a::Animal{S},b::Animal{S}) where S<:Species = true
EcosystemCore.mates(a::Agent, b::Agent) = false

gs = [Grass(id,max_size) for id in 1:n_grass]
ss = [Sheep(id,energy_sheep,Δenergy_sheep,sheep_reproduce, sheep_foodprob) for id in (n_grass+1):(n_grass+n_sheep)]
ws = [Wolf(id,energy_wolf,Δenergy_wolf,wolf_reproduce, wolf_foodprob) for id in (n_grass+n_sheep+1):(n_grass+n_sheep+n_wolves)]
as = vcat(gs,ss,ws)

w = World(as)

counts = Dict(n=>[c] for (n,c) in agent_count(w))
function _save(w::World)
    for (n,c) in agent_count(w)
        push!(counts[n],c)
    end
end

logcb = every_nth(w->(@info w), 1)
savecb = every_nth(_save, 1)
cbs = [logcb, savecb]

simulate!(w, 200, callbacks=cbs)

using Plots
plt = plot()
for (n,c) in counts
    plot!(plt, c, label="$n", lw=2)
end
savefig(plt, "docs/src/lecture_02/pred-prey.png")
# savefig(plt, "docs/src/lecture_03/pred-prey.png")
display(plt)
