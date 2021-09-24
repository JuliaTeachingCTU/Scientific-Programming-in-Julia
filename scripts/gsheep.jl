using EcosystemCore
using StatsBase
#gs = [Grass(id,5) for id in 1:5]
#ss = [Sheep(id,10.0,5.0,0.5,0.5) for id in 6:10]
#ws = [Wolf(id,20.0,10.,0.1,0.1) for id in 11:12]
#
#world = World(vcat(gs, ss, ws))
#simulate!(world, 10, callbacks=[w->@show w])


struct ⚥Sheep <: Animal
    sheep::Sheep
    sex::Symbol
end
⚥Sheep(id,E,ΔE,pr,pf,sex) = ⚥Sheep(Sheep(id,E,ΔE,pr,pf),sex)

EcosystemCore.id(g::⚥Sheep) = EcosystemCore.id(g.sheep)
EcosystemCore.energy(g::⚥Sheep) = energy(g.sheep)
EcosystemCore.energy!(g::⚥Sheep, ΔE) = energy!(g.sheep, ΔE)
EcosystemCore.reproduction_prob(g::⚥Sheep) = reproduction_prob(g.sheep)
EcosystemCore.food_prob(g::⚥Sheep) = food_prob(g.sheep)

EcosystemCore.eats(::⚥Sheep, ::Grass) = true
# eats(::⚥Sheep, ::PoisonedGrass) = true
EcosystemCore.eat!(s::⚥Sheep, g::Plant, w::World) = eat!(s.sheep, g, w)

mates(a::Plant, ::⚥Sheep) = false
mates(a::Animal, ::⚥Sheep) = false
mates(g1::⚥Sheep, g2::⚥Sheep) = g1.sex != g2.sex
function find_mate(g::⚥Sheep, w::World)
    ms = filter(a->mates(a,g), w.agents |> values |> collect)
    isempty(ms) ? nothing : sample(ms)
end

function EcosystemCore.reproduce!(s::⚥Sheep, w::World)
    m = find_mate(s,w)
    if !isnothing(m)
        energy!(s, energy(s)/2)
        vals = [getproperty(s.sheep,n) for n in fieldnames(Sheep) if n!=:id]
        new_id = w.max_id + 1
        ŝ = ⚥Sheep(new_id, vals..., rand(Bool) ? :female : :male)
        w.agents[EcosystemCore.id(ŝ)] = ŝ
        w.max_id = new_id
    end
end

f = ⚥Sheep(1,3.0,1.0,1.0,1.0,:female)
m = ⚥Sheep(2,4.0,1.0,1.0,1.0,:male)
w = World([f,m])
@show w
simulate!(w, 3, callbacks=[w->@show w])

