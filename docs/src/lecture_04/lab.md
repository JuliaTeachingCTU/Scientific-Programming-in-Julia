# Lab 4: Packages, Tests, Continuous Integration

In this lab you will practice common development workflows in Julia.
At the end of the lab you will have
- Your own package called `Ecosystem.jl` which you can conveniently install in any Julia REPL
- Tests for the major functionality of your package
- Set up continuous integration (CI) via Github Actions to automatically execute your tests

## Separating core and extended functionality

To practice working with multiple packages that build on top of each other we
first separate the core functionality of our ecosystem into a package called
`EcosystemCore.jl`. This core package [already
exists](https://github.com/JuliaTeachingCTU/EcosystemCore.jl) and defines the
interface to `Animal` and `Plant`. It also contains the three
basic species types `Grass`, `Sheep`, and `Wolf`, as well as the most important
functions: `eat!`, `agent_step!`, `find_food` and `reproduce!`, `model_step!`.

Your task is to create your own package `Ecosystem.jl` which will contain the
utitlity functions `simulate!`, `agent_count`, `every_nth` that we created,
as well as the new species `Mushroom`.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
1. Familiarize yourself with `EcosystemCore.jl`.

2. Create a new package by starting a julia REPL, typing `]` to enter the `Pkg`
   REPL and writing `generate Ecosystem`. This will create a new package called
   `Ecosystem` with a `Project.toml` and one file `src/Ecosystem.jl`

3. Navigate into the newly generated package folder via `;cd Ecosystem` and
   activate the environment via `]activate .`.

4. Add `EcosystemCore.jl` as a dependency by running
   ```
   ]add https://github.com/JuliaTeachingCTU/EcosystemCore.jl.git
   ```

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
You should now have only a single dependency in your new package:
```julia
(Ecosystem) pkg> st
     Project Ecosystem v0.1.0
      Status `~/Ecosystem/Project.toml`
  [3e0d8730] EcosystemCore v0.1.0 `https://github.com/JuliaTeachingCTU/EcosystemCore.jl.git#main`
```
You can try if `using EcosystemCore` correctly loads the core package and use
its exported functions and types as in the labs before.
```@repl lab04
using EcosystemCore
grass = Grass(1,5);
sheep = Sheep(2,10.0,5.0,0.1,0.1);
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
1. Next, let's add the utility functions `simulate!`, `agent_count`, and
   `every_nth`, as well as the new species `Mushroom` along with its necessary
   method overloads to the `Ecosystem` module.

   While you are adding functionality to your package you can make great use of
   `Revise.jl`.  Loading `Revise.jl` before your `Ecosystem.jl` will automatically
   recompile (and invalidate old methods!) while you develop.  You can install it
   in your global environment and and create a `USERDIR/.config/startup.jl` which always loads
   `Revise`. It can look like this:
   ```julia
   # try/catch block to make sure you can start julia if Revise should not be installed
   try
       using Revise
   catch e
       @warn(e.msg)
   end
   ```

2. Note that you either have to `import` a method to overload it or
   define your functions like `Ecosystem.function_name` e.g.:
   ```julia
   EcosystemCore.eats(::Animal{Sheep}, ::Plant{Mushroom}) = true
   ```
   which is often the preferred way of doing it.

3. Export all types and functions that should be accessible from outside your
   package.  This should include at least `agent_count`, `simulate!`,
   `every_nth`, probably at all species types, and the `World`.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
Now you can run one of your simulation scripts like below
```julia
# only load the Ecosystem package which depends on EcosystemCore
using Ecosystem

n_grass       = 500
regrowth_time = 17.0
# ...

world = World([...])

# ...
simulate!(world,100)
```
You can put your simulation scripts in the same package in a new folder called
`scripts` or `examples` if you like.
```@raw html
</p></details>
```

## Testing the Ecosystem

Every well maintained package should contain tests. In Julia the tests have to
be located in the `test` folder in the package root. The `test` folder
has to contain at least one file called `runtests.jl` which can `include` more
files. A minimal package structure can look like below.
```
.
├── Project.toml
├── README.md
├── src
│   └── Ecosystem.jl
└── test
    └── runtests.jl
```
To start testing we need Julia's `Test` package which you can install via `]add Test`.
We do not want `Test` in the dependencies of `Ecosystem.jl`, which we can achieve
by creating new `[extras]` and `[target]` sections in the `Project.toml`:
```
name = "Ecosystem"
uuid = "some-uuid"
authors = ["Your Name <yourname@email.com>"]
version = "0.1.0"

[deps]
EcosystemCore = "3e0d8730-8ea0-4ee2-afe6-c85384c618a2"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

With `Test.jl` as an extra dependency you can start writing your `test/runtests.jl` file.
```@example lab04
using Scientific_Programming_in_Julia # hide
using Test
# using Ecosystem  # in your code this line as to be uncommented ;)

@testset "agent_count" begin
    @test agent_count(Mushroom(1,1,1)) == 1
end
nothing # hide
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Create a `@testset` and fill it with tests for `agent_count` that cover all
of its four methods.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab04
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
nothing # hide
```
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
Pushing this file to your repository should result in Github automatically running
your julia tests.

You can add a status badge to your README by copying the status badge string
from the Github Actions tab.
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
1. Log into `codecov.io` with your Github user name and give codecov access to
   your new repository.

2. Add the codecov steps below to your `RunTests.yml`.  Note that **code coverage
   does not mean that your code is properly tested**!  It is simply measuring
   which lines have been hit during the execution of your tests, which does not
   mean that your code (or your tests) are correct.

3. Add the codecov status badge to your README.
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
