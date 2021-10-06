# Lab 4: Packages, Tests, Continuous Integration

# TODO
* demonstrate versioning


In this lab you will practice common development workflows in Julia.
At the end of the labe you will have
- Your own package called `Ecosystem.jl` which you can conveniently install in any Julia REPL
- Tests for the major functionality of your package
- Set up continuous integration (CI) via Github Actions to automatically execute your tests

## Separating core and extended functionality

To practice working with multiple packages that build on top of each other we
first separate the core functionality of our ecosystem into a package called
`EcosystemCore.jl`. This core package [already
exists](https://github.com/JuliaTeachingCTU/EcosystemCore.jl) and defines the
interface to `AbstractAnimal` and `AbstractPlant`. It also contains the three
basic types `Grass`, `Sheep`, and `Wolf`, as well as the most important
functions: `eat!`, `agent_step!`, `find_food` and `reproduce!`.

Your task is to create your own package `Ecosystem.jl` which will contain the
utitlity functions `simulate!`, `agent_count`, `every_nth` that we created,
as well as the new types `PoisonedGrass` and `⚥Sheep`.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
1. Create a new package by starting a julia REPL, typing `]` to enter the `Pkg` REPL and writing `generate Ecosystem`. This will create a new package called `Ecosystem` with a `Project.toml` and one file `src/Ecosystem.jl`

2. Exit julia, navigate into the newly created `Ecosystem` folder and restart julia in the `Ecosystem` environment by typing `julia --project`.

3. Add `EcosystemCore.jl` as a dependency by running
   ```
   ]add https://github.com/JuliaTeachingCTU/EcosystemCore.jl.git
   ```

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
You should now be able to run `using EcosystemCore` in your REPL to precomplie
the core package.
```@repl
using EcosystemCore
grass = Grass(1, 5.0, 5.0);          # id = 1
sheep = Sheep(2, 10.0, 5.0, 0.1, 0.1); # id = 2
world = World([grass, sheep])
eat!(sheep,grass,world);
world
```
```@raw html
</p></details>
```


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
1. Next, lets add the utility functions `simulate!`, `agent_count`, and `every_nth`, as well as the two new types `Mushroom` and `⚥Sheep` along with the necessary functions and method overloads. *TODO: update this*

2. Note that you either have to `import` a method to overload it or do something like this
   ```julia
   EcosystemCore.eats(::Sheep, ::PoisonedGrass) = true
   ```

3. Export all types and functions that should be accessible from outside your package.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
Partial solution that should reside inside the `Ecosystem` pkg
```julia
for S in (Sheep, Wolf)
  @eval begin
        EcosystemCore.mates(a::Animal{$S,Female}, b::Animal{$S,Male}) = true
        EcosystemCore.mates(a::Animal{$S,Male}, b::Animal{$S,Female}) = true
  end
end
EcosystemCore.mates(a::Agent, b::Agent) = false

using EcosystemCore: max_size # not exported from EcosystemCore

## this should be probably part of the EcosystemCore
species(::Plant{P}) where P <: PlantSpecies = P
species(::Animal{A}) where A <: AnimalSpecies = A
##

agent_count(p::Plant) = size(p)/max_size(p)
agent_count(::Animal) = 1
agent_count(as::Vector{<:Agent}) = sum(agent_count,as)

function agent_count(w::World)
    function op(d::Dict,a::A) where A<:Agent
        n = nameof(species(a))
        if n in keys(d)
            d[n] += agent_count(a)
        else
            d[n] = agent_count(a)
        end
        return d
    end
    foldl(op, w.agents |> values |> collect, init=Dict{Symbol,Real}())
end
```

In a fresh REPL you should now be able to run one of your simulation scripts
like below
```julia
# only load the Ecosystem package which depends on EcosystemCore
using Ecosystem

n_grass       = 500
regrowth_time = 17.0

n_sheep         = 100
Δenergy_sheep   = 5.0
sheep_reproduce = 0.5
sheep_foodprob  = 0.4

n_wolves       = 8
Δenergy_wolf   = 17.0
wolf_reproduce = 0.03
wolf_foodprob  = 0.02

gs = [Grass(id, regrowth_time) for id in 1:n_grass]
ss = [Sheep(id, 2*Δenergy_sheep, Δenergy_sheep, sheep_reproduce, sheep_foodprob) for id in n_grass+1:n_grass+n_sheep]
ws = [Wolf(id, 2*Δenergy_wolf, Δenergy_wolf, wolf_reproduce, wolf_foodprob) for id in n_grass+n_sheep+1:n_grass+n_sheep+n_wolves]

w = World(vcat(gs,ss,ws))

counts = Dict(n=>[c] for (n,c) in agent_count(w))
for _ in 1:100
    world_step!(w)
    @show agent_count(w)
end
```
```@raw html
</p></details>
```

## Testing the Ecosystem

Every well maintained package should contain tests. In Julia the tests have to
be located in the `test` folder of package root folder. The `test` folder
has to contain at least one file called `runtests.jl` which can `include` more
files. A minimal package structure can look like the `EcosystemCore` package
below.
```
.
├── Project.toml
├── README.md
├── src
│   └── EcosystemCore.jl
└── test
    └── runtests.jl
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Write at least one test for each function/method you added to your package.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
As an example of how to write tests in Julia you can take a look at the
[`runtests.jl`](https://github.com/JuliaTeachingCTU/EcosystemCore.jl/blob/main/test/runtests.jl) file of `EcosystemCore.jl`.
```@raw html
</p></details>
```

## Github & Continuous Integration

Another good standard is to use a versioning tool like `git`. It will save you
tons of headaches and you will never have to worry again that you could loose
some of your work.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Create a new repository called `Ecosystem.jl` on Github (public or private,
however you want). A good repo should also contain a `README.md` file which
briefly describes what it is about.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
You can turn a folder into a git repo by running `git init`. After you created
a new repo on Github you can connect your local repo to the empty Github repo
like below
```
# turn folder into git repo
git init
# point local repo to empty github repo
git remote add origin https://github.com/username/Ecosystem.jl.git
# rename master branch->main
git branch -M main
# create your initial commit
git add .
git commit -m "init commit"
# push your contents
git push -u origin main
```
```@raw html
</p></details>
```

As a last step, we will add a *Github Action* to your repository, which will
run your test on every commit to the `main` branch (or for every pull request).

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Create a file (in a hidden subdirectory) `.github/workflows/RunTests.yml`
with the contents below
```julia
name: Run tests

on:
  push:
    branches:
      - main
  pull_request:

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        julia-version: ['1.6']
        julia-arch: [x64]
        os: [ubuntu-latest]

    steps:
      - uses: actions/checkout@v2
      - uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.julia-version }}
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
```
Pushing this file to your repository should result in github automatically running
your julia tests.
```@raw html
</div></div>
```

## Code coverage

If you still have time you can add code coverage reports to your repository.
They will show you which parts of your repo have been covered by a test and which
have not. To get coverage reports you have to give `codecov.io` access to your
repository and add a few steps to your `RunTests.yml` that upload the coverage
report during the Github Action.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise (optional)</header>
<div class="admonition-body">
```
Add the codecov steps below to your `RunTests.yml` and get a code coverage
above 95%. Note that **code coverage does not mean that your code is properly
tested**!  It is simply measuring which lines have been hit during the
execution of your tests, which does not mean that your code (or your tests) are
correct.
```julia
steps:
  - uses: julia-actions/julia-processcoverage@v1
  - uses: codecov/codecov-action@v2
    with:
      file: lcov.info
```

```@raw html
</div></div>
```
