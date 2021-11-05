# Homework 4: More Unit Tests


In this homework you will finish writing your unit tests for your `Ecosystem.jl`.


## How to submit

The autoeval system expects you to upload a `.zip` file of your `Ecosystem`
with at least the following files
```
.
└── Ecosystem
    ├── Project.toml
    ├── src
    │   └── Ecosystem.jl
    └── test
        ├── every_nth.jl
        ├── mushroom.jl
        └── runtests.jl
```
The files `every_nth.jl` and `mushroom.jl` have to contain a single testset,
so for example, the `every_nth.jl` file should look like this:
```julia
@testset "every_nth" begin
    # your tests go in here
    @test ...
end
```
We will run those two testsets on our own (hopefully correct) implementation of
`Ecosystem.jl` and check if your tests pass. Subsequently we will run them on
an *incorrect* implementation of `Ecosystem.jl` and verify that your tests fail
with the incorrect package. Hence, you will get full points if your tests pass
on a correct implementation and fail on an incorrect implementation.

## Testing `every_nth`

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Compulsory Homework (1 point)</header>
<div class="admonition-body">
```
Create a `@testset` for `every_nth(f,n)` in the file `test/every_nth.jl`.
The testset should verify that the inner function `f` is called only every `n`
times. This testset must contain at least two `@test` calls. One should verify
that a variable *has not* changed the first `n-1` calls to `f`. The second one
should check that the variable *has* changed.
```@raw html
</div></div>
```

## Testing `Mushroom`

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Compulsory Homework (1 point)</header>
<div class="admonition-body">
```
Create a `@testset` for `Mushroom` that verifies that
`eat!(::Sheep,::Mushroom,::World)` does what we expect. Calling `eat!` on a
sheep and a mushroom should set the size of the mushroom to zero and decrease
the energy of the sheep appropriately.
```@raw html
</div></div>
```

## Code coverage

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise (optional)</header>
<div class="admonition-body">
```
Get your test coverage as computed by `codecov` (visible in your README badge)
above 95% by implementing a test for `simulate!`.
```@raw html
</div></div>
```
