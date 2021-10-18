using Scientific_Programming_in_Julia
using EcosystemCore
using Test

@testset "agent_count" begin
    grass1 = Grass(1,1,5)
    grass2 = Grass(2,2,5)
    sheep  = Sheep(3,1,1,1,1)
    wolf   = Wolf(5,2,2,2,2)
    world  = World([sheep,grass1,grass2,wolf])

    @test agent_count(grass1) ≈ 0.2
    @test agent_count(sheep) == 1
    @test agent_count([grass2,grass1]) ≈ 0.6
    res = agent_count(world)
    tst = Dict(:Sheep=>1,:Wolf=>1,:Grass=>0.6)
    for (k,_) in res
        @test res[k] ≈ tst[k]
    end
end

@testset "Mushroom" begin
    sheep = Sheep(5,10,2,1,1)
    mushr = Mushroom(2,5,5)
    world = World([sheep,mushr])

    EcosystemCore.eat!(sheep,mushr,world)
    @test size(mushr) == 0
    @test energy(sheep) == 0
end


@testset "every_nth" begin
    i = 0
    f() = i+=1
    cb = every_nth(f,3)

    cb(); cb();
    @test i == 0
    cb()
    @test i == 1
end
