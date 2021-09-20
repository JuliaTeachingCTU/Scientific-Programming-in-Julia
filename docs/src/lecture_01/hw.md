# Homework 1: Extending `polynomial` the other way
```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (1+1 points)</header>
<div class="admonition-body">
```

Extend the original polynomial function to the case where `x` is a square matrix. Create a function called `circlemat`, that returns `nxn` matrix $$A(n)$$ with the following elements
```math
\left[A(n)\right]_{ij} = 
\begin{cases}
   1 &\text{if } (i = j-1 \land j > 1) \lor (i = n \land j=1) \\
   1 &\text{if } (i = j+1 \land j < n) \lor (i = 1 \land j=n) \\
   0 & \text{  otherwise}
\end{cases}
```
and evaluate the polynomial
```math
f(A) = I + A + A^2 + A^3.
```
, at point $$A = A(10)$$.

**HINTS** for matrix definition:
You can try one of these options:
- create matrix with all zeros with `zeros(n,n)`, use two nested for loops going in ranges `1:n` and if condition with logical or `||`, and `&&` 
- employ array comprehension with nested loops `[expression for i in 1:n, j in 1:n]` and ternary operator `condition ? true branch : false`

**HINTS** for `polynomial` extension:
- extend the original example (one with for-loop) to initialize the `accumulator` variable with matrix of proper size (use `size` function to get the dimension), using argument typing for `x` is preferred to distinguish individual implementations `<: AbstractMatrix`
or
- test later defined `polynomial` methods, that may work out of the box

```@raw html
</div></div>
<details class = "solution-body" hidden>
<summary class = "solution-header">Solution:</summary><p>
```

As always there are multiple options for the `circlemat` function definition. Here we show the two, that we have foreshadowed in the homework hints:

- longer version using incremental definition in for loop
```@repl 1
function circlemat(n)
    A = zeros(Int, n, n) # creates nxn matrix of zeros
    for i in 1:n
        for j in 1:n
            if (i == j-1 && j > 1) || (i == n && j == 1) || (i == j+1 && j < n) || (i == 1 && j == n)
                A[i,j] = 1
            end
        end
    end
    return A
end
circlemat(10)
```

- short version with comprehension and ternary operator
```@repl 1
circlemat(n) = [(i == j-1 && j > 1) || (i == n && j == 1) || (i == j+1 && j < n) || (i == 1 && j == n) ? 1 : 0 for i in 1:n, j in 1:n]
circlemat(10)
```
Both version should give the same answer.

Extending the original `polynomial` to matrix valued point `x = A`, requires only small changes to the initialization of `accumulator` variable. Running directly the original code fails on `MethodError`, because julia cannot add a matrix to an integer.
```@repl 1
function polynomial(a, x::AbstractMatrix) # we are limiting this function to everything that is subtype of AbstractMatrix
    accumulator = zeros(eltype(x), size(x)) # zeros of the same type and size as `x`
    for i in length(a):-1:1
        accumulator += x^(i-1) * a[i] # ! 1-based indexing for arrays
    end
    return accumulator
end
```

```@repl 1
A = circlemat(10) # matrix of size 10x10
coeffs = ones(4)  # coefficients of polynomial
polynomial(coeffs, A)
```

The other option is to use the more abstract version that we have defined to work with generators/iterators. For example
```@repl generator
polynomial(a, x) = sum(ia -> x^(ia[1]-1) * ia[2], enumerate(a))
```
works out of the box
```@repl generator
circlemat(n) = [(i == j-1 && j > 1) || (i == n && j == 1) || (i == j+1 && j < n) || (i == 1 && j == n) ? 1 : 0 for i in 1:n, j in 1:n] #hide
A = circlemat(10) #hide
coeffs = ones(4)  #hide
polynomial(coeffs, A)
```

```@raw html
</p></details>
```


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise (voluntary)</header>
<div class="admonition-body">
```
Install `GraphRecipes` and `Plots` packages into our `./L1Env/` environment defined during the lecture and figure out, how to plot the graph defined by adjacency matrix `A` from the homework.

**HINTS**:
- There is help command inside the the pkg mod of the REPL. Type `? add` to find out how to install a package. Note that both pkgs are registered.
- Follow a guide in the `Plots` pkg's documentation, which is accessible through `docs` icon on top of the README in the GitHub [repository](https://github.com/JuliaPlots/Plots.jl). Direct [link](http://docs.juliaplots.org/latest/graphrecipes/introduction/#GraphRecipes).

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

Activate `L1Env` environment in pkg mode, if it is not currently active.
```julia
pkg> activate ./L1Env
```
Installing pkgs is achieved using the `add` command. Running `] ? add` returns a short piece of documentation for this command:
```julia
pkg> ? add
[...]
  Examples

  pkg> add Example                                          # most commonly used for registered pkgs (installs usually the latest release)
  pkg> add Example@0.5                                      # install with some specific version (realized through git tags)
  pkg> add Example#master                                   # install from master branch directly
  pkg> add Example#c37b675                                  # install from specific git commit
  pkg> add https://github.com/JuliaLang/Example.jl#master   # install from specific remote repository (when pkg is not registered)
  pkg> add git@github.com:JuliaLang/Example.jl.git          # same as above but using the ssh protocol
  pkg> add Example=7876af07-990d-54b4-ab0e-23690620f79a     # when there are multiple pkgs with the same name
```
As the both `Plots` and `GraphRecipes` are registered and we don't have any version requirements, we will use the first option.
```julia
pkg> add Plots
pkg> add GraphRecipes
```
This process downloads the pkgs and triggers some build steps, if for example some binary dependencies are needed. The process duration depends on the "freshness" of Julia installation and the size of each pkg. With `Plots` being quite dependency heavy, expect few minutes. After the installation is complete we can check the updated environment with the `status` command.
```julia
pkg> status
```

The plotting itself as easy as calling the `graphplot` function on our adjacency matrix.
```@repl 1
using GraphRecipes, Plots
graphplot(A)
```
```@example 1
graphplot(A) #hide
```

```@raw html
</p></details>
```

# How to submit?
The guide is located [here](@ref homeworks).
