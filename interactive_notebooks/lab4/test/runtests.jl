using Test
using Ecosystem

@testset "World iteration interface" begin
    world = World([Grass(1, 2, 2), Sheep(2), Wolf(3)])

    @test length(world) == 3
    @test eltype(world) === Agent
    @test Set(collect(world)) == Set(values(world.agents))
end

@testset "Diet trait" begin
    sheep = Sheep(1)
    wolf = Wolf(2)
    grass = Grass(3, 2, 2)

    @test eats(wolf, sheep)
    @test eats(sheep, grass)
    @test !eats(sheep, wolf)
    @test !eats(wolf, wolf)
end

@testset "Total animal energy" begin
    @test total_energy(World([Grass(1, 2, 2), Sheep(2; E=3.0)])) == 3.0
    @test total_energy(World([Grass(1, 2, 2)])) == 0.0
end
