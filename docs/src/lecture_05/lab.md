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
sheep1 = Sheep(1,s=female)
sheep2 = Sheep(3002,s=male)
world = World(vcat([sheep1,sheep2], [Grass(i) for i=2:3000]))
@code_warntype find_food(sheep1, world)
```
```@repl unstable
@btime find_food($sheep1, $world)
@btime reproduce!($sheep1, $world)
```

## More Stable `Ecosystem`
```@example morestable
sheep1 = Sheep(1,s=female)
sheep2 = Sheep(3002,s=male)
world = World(vcat([sheep1,sheep2], [Grass(i) for i=2:3000]))
@code_warntype find_food(sheep1, world)
```
```@repl morestable
@btime find_food($sheep1, $world)
@btime reproduce!($sheep1, $world)
```


## Stable `Ecosystem`
```@example stable
sheep1 = Sheep(1,S=Female)
sheep2 = Sheep(3002,S=Male)
world = World(vcat([sheep1,sheep2], [Grass(i) for i=2:3000]))
@code_warntype find_food(sheep1, world)
```
```@repl stable
@btime find_food($sheep1, $world)
@btime reproduce!($sheep1, $world)
```

