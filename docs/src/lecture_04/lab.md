## Warmup

```@setup block
projdir = dirname(Base.active_project())
include(joinpath(projdir,"src","lecture_03","Lab03Ecosystem.jl"))
```

### Agents looking for food

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
Implement a method `find_food(a::Animal, w::World)` returns one randomly chosen
agent from all `w.agents` that can be eaten by `a` or `nothing` if no food could
be found. This means that if e.g. the animal is a `Wolf` you have to return one
random `Sheep`, etc.

*Hint*: You can write a general `find_food` method for all animals and move the
parts that are specific to the concrete animal types to a separate function.
E.g. you could define a function `eats(::Wolf, ::Sheep) = true`, etc.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example block
using StatsBase

function find_food(a::Animal, w::World)
    as = filter(x -> eats(a,x), w.agents |> values |> collect)
    isempty(as) ? nothing : sample(as)
end

eats(::Animal{Sheep},g::Plant{Grass}) = g.size > 0
eats(::Animal{Wolf},::Animal{Sheep}) = true
eats(::Agent,::Agent) = false
nothing # hide
```
```@raw html
</p></details>
```
```@repl block
sheep = Sheep(1)
world = World([Grass(2), sheep])
find_food(sheep, world)
```

### Stepping through time

We now have all necessary functions in place to make agents perform one step
of our simulation.  At the beginning of each step an animal looses energy.
Afterwards it tries to find some food, which it will subsequently eat. If the
animal then has less than zero energy it dies and is removed from the world. If
it has positive energy it will try to reproduce.

Plants have a simpler life. They simply grow if they have not reached their maximal size.
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
1. Implement a method `agent_step!(::Animal,::World)` which performs the following steps:
    - Decrement $E$ of agent by `1.0`.
    - With $p_f$, try find some food and eat it.
    - If $E<0$, the animal dies.
    - With $p_r$, try to reproduce.
2. Implement a method `agent_step!(::Plant,::World)` which performs the following steps:
    - If the size of the plant is smaller than `max_size`, increment the plant's size by one.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example block
function agent_step!(p::Plant, w::World)
    if p.size < p.max_size
        p.size += 1
    end
end

function agent_step!(a::Animal, w::World)
    a.energy -= 1
    if rand() <= a.foodprob
        dinner = find_food(a,w)
        eat!(a, dinner, w)
    end
    if a.energy < 0
        kill_agent!(a,w)
        return
    end
    if rand() <= a.reprprob
        reproduce!(a,w)
    end
end

# make it possible to eat nothing
eat!(::Animal, ::Nothing, ::World) = nothing
nothing # hide
```
```@raw html
</p></details>
```
An `agent_step!` of a sheep in a world with a single grass should make it consume the grass,
let it reproduce, and eventually die if there is no more food and its energy is at zero:
```@repl block
sheep = Sheep(1,2.0,2.0,1.0,1.0);
grass = Grass(2,2,2);
world = World([sheep, grass])
agent_step!(sheep, world); world
# NOTE: The second agent step leads to an error.
# Can you figure out what is the problem here?
agent_step!(sheep, world); world
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
Finally, lets implement a function `world_step!` which performs one
`agent_step!` for each agent.  Note that simply iterating over all agents could
lead to problems because we are mutating the agent dictionary.  One solution for
this is to iterate over a copy of all agent IDs that are present when starting
to iterate over agents.  Additionally, it could happen that an agent is killed
by another one before we apply `agent_step!` to it. To solve this you can check
if a given ID is currently present in the `World`.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example block
function world_step!(world::World)
    # make sure that we only iterate over IDs that already exist in the
    # current timestep this lets us safely add agents
    ids = copy(keys(world.agents))

    for id in ids
        # agents can be killed by other agents, so make sure that we are
        # not stepping dead agents forward
        !haskey(world.agents,id) && continue

        a = world.agents[id]
        agent_step!(a,world)
    end
end
```
```@raw html
</p></details>
```
```@repl block
w = World([Sheep(1), Sheep(2), Wolf(3)])
world_step!(w); w
world_step!(w); w
world_step!(w); w
```

Finally, lets run a few simulation steps and plot the solution
```@example block
n_grass  = 1_000
n_sheep  = 40
n_wolves = 4

gs = [Grass(id) for id in 1:n_grass]
ss = [Sheep(id) for id in (n_grass+1):(n_grass+n_sheep)]
ws = [Wolf(id) for id in (n_grass+n_sheep+1):(n_grass+n_sheep+n_wolves)]
w  = World(vcat(gs,ss,ws))

counts = Dict(n=>[c] for (n,c) in agent_count(w))
for _ in 1:100
    world_step!(w)
    for (n,c) in agent_count(w)
        push!(counts[n],c)
    end
end

using Plots
plt = plot()
tolabel(::Type{Animal{Sheep}}) = "Sheep"
tolabel(::Type{Animal{Wolf}}) = "Wolf"
tolabel(::Type{Plant{Grass}}) = "Grass"
for (A,c) in counts
    plot!(plt, c, label=tolabel(A), lw=2)
end
plt
```


## Package: `Ecosystem.jl`

create your own package where you can organize the code that we have written so far

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
`]add PkgTemplates` to your global julia env and create a new package by running:
```julia
using PkgTemplates
Template(interactive=true)("Ecosystem")
```
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
This should have created a new folder `Ecosystem` which looks like below.
```
.
├── LICENSE
├── Manifest.toml
├── Project.toml
├── README.md
├── src
│   └── Ecosystem.jl
└── test
    └── runtests.jl
```
If you `]activate ~/path/to/Ecosystem` you should be able to run `]test` to run the autogenerated test (which is not doing anything)
and get the following output:
```
(Ecosystem) pkg> test
     Testing Ecosystem
      Status `/private/var/folders/6h/l9_skfms2v3dt8z3zfnd2jr00000gn/T/jl_zd5Uai/Project.toml`
  [e77cd98c] Ecosystem v0.1.0 `~/repos/Ecosystem`
  [8dfed614] Test `@stdlib/Test`
      Status `/private/var/folders/6h/l9_skfms2v3dt8z3zfnd2jr00000gn/T/jl_zd5Uai/Manifest.toml`
  [e77cd98c] Ecosystem v0.1.0 `~/repos/Ecosystem`
  [2a0f44e3] Base64 `@stdlib/Base64`
  [b77e0a4c] InteractiveUtils `@stdlib/InteractiveUtils`
  [56ddb016] Logging `@stdlib/Logging`
  [d6f4376e] Markdown `@stdlib/Markdown`
  [9a3f8284] Random `@stdlib/Random`
  [ea8e919c] SHA v0.7.0 `@stdlib/SHA`
  [9e88b42a] Serialization `@stdlib/Serialization`
  [8dfed614] Test `@stdlib/Test`
     Testing Running tests...
Test Summary: |Time
Ecosystem.jl  | None  0.0s
     Testing Ecosystem tests passed 
```
```@raw html
</p></details>
```


!!! warning
    From now on make sure that you **always** have the `Ecosystem` enviroment
    enabled. Otherwise you will not end up with the correct dependencies in your
    packages
    
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
Next, let's add the types and functions we have defined so
far. You can use `include("path/to/file.jl")` in the main module file at
`src/Ecosystem.jl` to bring some structure in your code. An exemplary
file structure could look like below.
```
.
├── LICENSE
├── Manifest.toml
├── Project.toml
├── README.md
├── src
│   ├── Ecosystem.jl
│   ├── animal.jl
│   ├── plant.jl
│   └── world.jl
└── test
    └── runtests.jl
```
While you are adding functionality to your package you can make great use of
`Revise.jl`.  Loading `Revise.jl` before your `Ecosystem.jl` will automatically
recompile (and invalidate old methods!) while you develop.  You can install it
in your global environment and and create a `$HOME/.config/startup.jl` which always loads
`Revise`. It can look like this:
```julia
# try/catch block to make sure you can start julia if Revise should not be installed
try
    using Revise
catch e
    @warn(e.msg)
end
```
```@raw html
</div></div>
```

!!! warning
    At some point along the way you should run into problems with the `sample`
    functions or when trying `using StatsBase`. This is normal, because you have
    not added the package to the `Ecosystem` environment yet. Adding it is as easy
    as `]add StatsBase`. Your `Ecosystem` environment should now look like this:
    ```
    (Ecosystem) pkg> status
    Project Ecosystem v0.1.0
    Status `~/repos/Ecosystem/Project.toml`
      [2913bbd2] StatsBase v0.33.21
    ```


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
write tests

export stuff
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@raw html
</p></details>
```


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise:</header>
<div class="admonition-body">
```
CI Tests + Codecov
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@raw html
</p></details>
```
