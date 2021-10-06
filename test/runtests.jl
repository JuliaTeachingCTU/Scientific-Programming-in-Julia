using Scientific_Programming_in_Julia
using EcosystemCore
using Test

@testset "World" begin
    w = World(
        (Mushroom, 1, (10,)),
        (Sheep, 1, (2,1,0.1,0.2))
    )
    m = EcosystemCore.find_rand(x->isa(x,Plant),w)
    s = EcosystemCore.find_rand(x->isa(x,Animal),w)
    @test m.max_size == 10
    @test s.energy == 2
    @test s.Δenergy == 1
    @test s.reprprob == 0.1
    @test s.foodprob == 0.2

    w = World(
        (Mushroom, 10, (max_size=10,)),
        (Sheep, 20, (Δenergy=1,energy=2,foodprob=1,reprprob=0.5))
    )
    m = EcosystemCore.find_rand(x->isa(x,Plant),w)
    s = EcosystemCore.find_rand(x->isa(x,Animal),w)
    @test m.max_size == 10
    @test s.energy == 2
    @test s.Δenergy == 1
    @test s.reprprob == 0.5
    @test s.foodprob == 1
end

@testset "agent_count" begin
    g = Grass(1,2,4)
    s = Sheep(2,1,1,1,1)
    w = Wolf(3,1,1,1,1)
    @test agent_count(g) == 0.5
    @test agent_count([Grass(1,2,2),Grass(2,1,2),Grass(3,0,2)]) == 1.5
    @test agent_count([Sheep(1,1,1,1,1)]) == 1
    @test agent_count(World([w,s,g,Grass(4,2,2)])) == Dict(:Wolf=>1, :Grass=>1.5, :Sheep=>1)
end

@testset "every_nth" begin
    i = 0
    f(w::World) = i+=1
    cb = every_nth(f,3)

    w = World(Dict{Int,Agent}(),1)
    cb(w); cb(w);
    @test i == 0
    cb(w)
    @test i == 1
end
