# Homework 6: Find variables
Following the lab exercises, you may thing that metaprogramming is a fun little exercise. Let's challenge this notion in this homework, where *YOU* are being trusted with catching all the edge cases in an AST.

## How to submit?
Put the code of the compulsory task inside `hw.jl`. Zip only this file (not its parent folder) and upload it to BRUTE. Your file should not use any 3rd party dependency.

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (2 points)</header>
<div class="admonition-body">
```
Your task is to find all single letter variables in an expression, i.e. for example when given expression
```julia
x + 2*y*z - c*x
```
return an array of *unique alphabetically sorted symbols* representing variables in an expression.
```julia
[:c, :x, :y, :z]
```
Implement this in a function called `find_variables`. Note that there may be some edge cases that you may have to handle in a special way, such as 
- variable assignments `r = x*x` should return the variable on the left as well (`r` in this case)
- ignoring symbols representing single letter function calls such as `f(x)`

```@raw html
</div></div>
<details class = "solution-body" hidden>
<summary class = "solution-header">Solution:</summary><p>
```

Nothing to see here.


```@raw html
</p></details>
```

# Voluntary exercise
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Voluntary exercise</header>
<div class="admonition-body">
```
Create a function that replaces each of `+`, `-`, `*` and `/` with the respective checked operation, which checks for overflow. E.g. `+` should be replaced by `Base.checked_add`.

```@raw html
</div></div>
<details class = "solution-body" hidden>
<summary class = "solution-header">Solution:</summary><p>
```
Not yet published.

```@raw html
</p></details>
```