# Homework 4: More Unit Tests

In this lab you will create another helper function for your new package `Ecosystem.jl`
and unit test it.

The more types of species we want to add to our simulation, the more tedious it
becomes to assign the correct IDs when creating the world. Hence, we will add a
mechanism that automatically assigns correct IDs to created agents.
```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (1 point)</header>
<div class="admonition-body">
```
Overload the constructor of `World` to accept an arbitrary number of configuration
tuples like
```julia
grass_config = (Grass, 200, (max_size = 10,))
sheep_config = (Sheep, 10, (Δenergy = 0.2, energy = 4.0, reprprob = 0.8, foodprob = 0.6))
```
and automatically create the desired agents with correct IDs.

```@raw html
</div></div>
```
The correct overload of the `World` constructor should result in behaviour like
below
```@example hw04
using Scientific_Programming_in_Julia
using EcosystemCore

grass_config = (Grass, 2, (max_size = 10,))
sheep_config = (Sheep, 1, (Δenergy = 0.2, energy = 4.0, reprprob = 0.8, foodprob = 0.6))
wolf_config  = (Wolf, 1, (energy = 10.0, Δenergy = 8.0, reprprob = 0.1, foodprob = 0.2))
World(grass_config, sheep_config, wolf_config)
```

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (1 point)</header>
<div class="admonition-body">
```
Write at least 4 unit tests that make sure that the correct number of agents is
produced, and that the arguments to the `Species` constructors are passed in
the right order.
```@raw html
</div></div>
```
