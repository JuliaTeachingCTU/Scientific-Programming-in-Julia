# [Homework 7: Creating world in 3 days/steps](@id hw07)

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (2 points)</header>
<div class="admonition-body">
```
Create a macro `@ecosystem` that should be able to define a world given a list of statements `@add # $species ${optional:sex}`
```julia
world = @ecosystem begin
    @add 10 Sheep Female    # adds 10 female sheep
    @add 2 Sheep Male       # adds 2 male sheep
    @add 100 Grass          # adds 100 pieces of grass
    @add 3 Wolf             # adds 5 wolf with random sex
end
```
`@add` should not be treated as a macro, but rather just as a syntax, that can be easily matched.

As this is not a small task let's break it into 3 steps
1. Define method `default_config(::Type{T})` for each `T` in `Grass, Wolf,...`, which returns a tuple of default parameters for that particular agent.
2. Define method `_add_agents(max_id, count::Int, species::Type{<:Species})` and `_add_agents(max_id, count::Int, species::Type{<:AnimalSpecies}, sex::Type{<:Sex})` that return an array of `count` agents of species `species` with `id` going from `max_id+1` to `max_id+count`. Default parameters should be constructed with `default_config`.
3. Define the underlying function `_ecosystem(ex)`, which parses the block expression and creates a piece of code that constructs the world.

You can test the macro (more precisely the `_ecosystem` function) with the following expression
```julia
ex = :(begin
    @add 10 Sheep Female
    @add 2 Sheep Male
    @add 100 Grass
    @add 3 Wolf
end)
genex = _ecosystem(ex)
world = eval(genex)
```

```@raw html
</div></div>
<details class = "solution-body" hidden>
<summary class = "solution-header">Solution:</summary><p>
```

Nothing to see here

```@raw html
</p></details>
```