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
ss = [Sheep(2*Δenergy_sheep,Δenergy_sheep,sheep_reproduce, sheep_foodprob) for _ in 1:n_sheep]
ws = [Wolf(2*Δenergy_wolf,Δenergy_wolf,wolf_reproduce, wolf_foodprob) for _ in 1:n_wolves]
as = vcat(gs,ss,ws)

w = World(as)

using Plots
ns = [:Wolf, :Sheep, :Grass]
counts = Dict(n=>[] for n in ns)
for i in 1:200
    for a in w.agents
        agent_step!(a,w)
    end
    cs = agent_count(w.agents)
    for (n,c) in cs
        push!(counts[n], c)
    end

    if mod(i,10) == 0
        plt = plot()
        for n in ns
            plot!(plt, counts[n], label="$(counts[n][end]) - $n")
        end
        display(plt)
    end
end
plt = plot()
for n in ns
    plot!(plt, counts[n], label="$n", lw=2)
end
# savefig(plt, "docs/src/lecture_02/pred-prey.png")
# savefig(plt, "docs/src/lecture_03/pred-prey.png")
display(plt)
