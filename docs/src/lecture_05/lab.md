## Unstable `Ecosystem`

```@setup unstable
projdir = dirname(Base.active_project())
include(joinpath(projdir,"src","lecture_04","Lab04Ecosystem.jl"))
using BenchmarkTools
using InteractiveUtils: @code_warntype
```
```@setup block
projdir = dirname(Base.active_project())
include(joinpath(projdir,"src","lecture_05","Lab05Ecosystem.jl"))
using BenchmarkTools
using InteractiveUtils: @code_warntype
```

```@example unstable
sheep = Sheep(1)
world = World(vcat([sheep], [Grass(i) for i=2:3000]))
@code_warntype find_food(sheep, world)
```
```@repl unstable
@btime find_food($sheep, $world)
```

## Stable `Ecosystem`
```@example block
sheep = Sheep(1)
world = World(vcat([sheep], [Grass(i) for i=2:3000]))
@code_warntype find_food(sheep, world)
```
```@repl block
@btime find_food($sheep, $world)
```
