# Homework 4: More Unit Tests


In this lab you will finish writing your unit tests for your `Ecosystem.jl`.

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (1 point)</header>
<div class="admonition-body">
```
Write a `@testset` for `every_nth(f,n)`. The testset should verify that the
inner function `f` is called only every `n` times. This testset must contain at
least two `@test` calls. One could verify that a variable *has not* changed the
first `n-1` calls to `f`. The second one can check that the variable *has*
changed.
```@raw html
</div></div>
```

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (1 point)</header>
<div class="admonition-body">
```
Create a `@testset` for `Mushroom` to verify that its size is set to zero and
that a `Sheep` looses energy if it eats a `Mushroom`.
```@raw html
</div></div>
```

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
