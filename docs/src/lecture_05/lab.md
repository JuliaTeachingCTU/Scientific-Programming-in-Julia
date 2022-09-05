## Unstable `Ecosystem`

```@setup unstable
projdir = dirname(Base.active_project())
include(joinpath(projdir,"src","lecture_04","Lab04Ecosystem.jl"))
using BenchmarkTools
using InteractiveUtils: @code_warntype
```
```@setup morestable
projdir = dirname(Base.active_project())
include(joinpath(projdir,"src","lecture_05","Lab05Ecosystem_Dict_Field.jl"))
using BenchmarkTools
using InteractiveUtils: @code_warntype
```
```@setup stable
projdir = dirname(Base.active_project())
include(joinpath(projdir,"src","lecture_05","Lab05Ecosystem_NamedTuple_Parametric.jl"))
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
@btime reproduce!($sheep, $world)
```

## More Stable `Ecosystem`
```@example morestable
sheep = Sheep(1)
world = World(vcat([sheep], [Grass(i) for i=2:3000]))
@code_warntype find_food(sheep, world)
```
```@repl morestable
@btime find_food($sheep, $world)
@btime reproduce!($sheep, $world)
```


## Stable `Ecosystem`
```@example stable
sheep = Sheep(1)
world = World(vcat([sheep], [Grass(i) for i=2:3000]))
@code_warntype find_food(sheep, world)
```
```@repl stable
@btime find_food($sheep, $world)
@btime reproduce!($sheep, $world)
```
