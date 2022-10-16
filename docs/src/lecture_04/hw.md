# Homework 4

In this homework you will have to write two additional `@testset`s for the
Ecosystem.  One testset should be contained in a file `test/sheep.jl` and verify
that the function `eat!(::Animal{Sheep}, ::Plant{Grass}, ::World)` works correctly.  Another
testset should be in the file `test/wolf.jl` and veryfiy that the function
`eat!(::Animal{Wolf}, ::Animal{Sheep}, ::World)` works correctly.

## How to submit?

Zip the whole package folder `Ecosystem.jl` and upload it to BRUTE.
The package has to include at least the following files:

```
├── src
│   └── Ecosystem.jl
└── test
    ├── sheep.jl  # contains only a single @testset
    ├── wolf.jl   # contains only a single @testset
    └── runtests.jl
```
Thet `test/runtests.jl` file can look like this:
```
using Test
using Ecosystem

include("sheep.jl")
include("wolf.jl")
# ...
```

## Test `Sheep`

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework:</header>
<div class="admonition-body">
```
1. Create a `Sheep` with food probability $p_f=1$
2. Create *fully grown* `Grass` and a `World` with the two agents.
3. Execute `eat!(::Animal{Sheep}, ::Plant{Grass}, ::World)`
4. `@test` that the size of the `Grass` now has `size == 0`
```@raw html
</div></div>
```


## Test `Wolf`

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework:</header>
<div class="admonition-body">
```
1. Create a `Wolf` with food probability $p_f=1$
2. Create a `Sheep` and a `World` with the two agents.
3. Execute `eat!(::Animal{Wolf}, ::Animal{Sheep}, ::World)`
4. `@test` that the World only has one agent left in the agents dictionary
```@raw html
</div></div>
```
