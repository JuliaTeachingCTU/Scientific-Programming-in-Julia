# [Lab 06: Digging deeper part 1](@id introspection_lab)
In this lab we are first going to inspect some tooling to help you understand what Julia does under the hood such as:
- looking at the code at different levels
- understanding what method is being called
- where to find the source code
- showing different levels of code optimization

Secondly we will start playing with the metaprogramming side of Julia, mainly covering:
- how to view abstract syntax tree (AST) of Julia code
- how to manipulate AST

These topics will be extended in the next [lecture](@ref macro_lecture)/[lab](@ref macro_lab), where we are going use metaprogramming to achieve greater good with macros and generated functions, which lie at the core of some state of the art Julia pkgs.

We will be again a little getting ahead of ourselves as we are going to use quite a few macros, which will be properly explained in the nextr lecture as well, however for now the important thing to know is that macro is just a special function, that accepts as an argument Julia code, which it modifies.

## Introspection
Let's start with the topic of code inspection, e.g. we may ask the following: What happens when I execute this
```julia
[i for i in 1:10]
```
We may be tempted to think, that this is some function call, for which we can call the `@which` macro to infer what is being called under the hood, however when we do this Julia will warn us that this is something "more complex"
```@repl lab06_intro
using InteractiveUtils #hide
@which [i for i in 1:10]
```
and that we should rather call the `Meta.@lower` macro, which spits out the so called lowered code of the expression.
```@repl lab06_intro
Meta.@lower [i for i in 1:10]
```

## Type introspection

```@repl lab06_intro
using EcosystemCore
dump(Animal)
```



### Optimization of broadcasting as seen from the different levels



## Calling LLVM/C/Fortran/(C++)/Pointers?


## Let's do some metaprogramming
Julia is so called homoiconic language, as it allows the language to reason about its code. This capability is inspired by years of development in other languages such as Lisp, Clojure or Prolog.

There are two easy ways to extract/construct the code structure [^1]
[^1]: Once you understand the recursive structure of expressions, the AST can be constructed manually like any other type.
- parsing code stored in string with internal `Meta.parse`
```@repl lab06_meta
code_parse = Meta.parse("x = 2")    # for single line expressions (additional spaces are ignored)
code_parse_block = Meta.parse("""
begin
    x = 2
    y = 3
    x + y
end
""") # for multiline expressions
```
- constructing an expression using `quote ... end` or  simple `:()` syntax
```@repl lab06_meta
code_expr = :(x = 2)    # for single line expressions (additional spaces are ignored)
code_expr_block = quote
    x = 2
    y = 3
    x + y   
end # for multiline expressions
```
Results can be stored into some variables, which we can inspect further.
```@repl lab06_meta
typeof(code_parse)
dump(code_parse)
```
```@repl lab06_meta
typeof(code_parse_block)
dump(code_parse_block)
```
The type of both multiline and single line expression is `Expr` with fields `head` and `args`. Notice that `Expr` type is recursive in the `args`, which can store other expressions resulting in a tree structure - abstract syntax tree (AST) - that can be visualized for example with the combination of `GraphRecipes` and `Plots` packages. 

```@example lab06_meta
using GraphRecipes
using Plots
plot(code_expr_block, fontsize=12, shorten=0.01, axis_buffer=0.15, nodeshape=:rect)
```

This recursive structure has some major performance drawbacks, because the `args` field is of type `Any` and therefore modifications of this expression level AST won't be type stable. Building blocks of expressions are `Symbol`s and literal values (numbers).


A possible nuisance of working with multiline expressions is the presence of `LineNumber` nodes, which can be removed with `Base.remove_linenums!` function.
```@repl lab06_meta
Base.remove_linenums!(code_parse_block)
```

Parsed expressions can be evaluate using `eval` function. 
```@repl lab06_meta
eval(code_parse)    # evaluation of :(x = 2) 
x                   # should be defined
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Before doing anything more fancy let's start with some simple manipulation of ASTs.
- Define a variable `code` to be as the result of parsing the string `"j = i^2"`. 
- Copy code into a variable `code2`. Modify this to replace the power `2` with a power `3`. Make sure that the original code variable is not also modified. 
- Copy `code2` to a variable `code3`. Replace `i` with `i + 1` in `code3`.
- Define a variable `i` with the value `4`. Evaluate the different code expressions using the `eval` function and check the value of the variable `j`.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@repl lab06_meta
code = Meta.parse("j = i^2")
code2 = copy(code)
code2.args[2].args[3] = 3
code3 = copy(code2)
code3.args[2].args[2] = :(i + 1)
i = 4
eval(code), eval(code2), eval(code3)
```

```@raw html
</p></details>
```

As the previous exercise has been focused on substitution of a single instance of expression. Let's try doing similar stuff in more general way.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Write function `substitute!(ex::Expr)` which replaces all `x` with `y`. Test this on the following expression
```@example lab06_meta
ex = :(x + x*x + y*x - sin(z))
nothing #hide
```

**HINT**: Use recursion because we are working with a recursive structure.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@repl lab06_meta
using Test #hide
function substitute!(ex)
    args = ex.args
    
    for i in 1:length(args)
        if args[i] == :x
            args[i] = :y
        
        elseif args[i] isa Expr
            substitute!(args[i])
        end
    end
    
    return ex
end

@test substitute!(ex) == :(y + y*y + y*y - sin(z))
```

```@raw html
</p></details>
```

Manipulating expressions allows us to create a wide variety of exercises, so here goes another, whose result may be more useful in the near future.
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Write function `wrap!(ex::Expr)` which wraps literal values (numbers) with a call to `f()`. You can test it on the following example
```@example lab06_meta
f = x -> convert(Float64, x)
ex = :(x*x + 2*y*x + y*y)
nothing #hide
```

**HINTS**:
- use recursion
- dispatch on `::Number` to detect numbers in an expression
- for testing purposes, create a copy of `ex` before mutating

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@repl lab06_meta
function wrap!(ex::Expr)
    args = ex.args
    
    for i in 1:length(args)
        args[i] = wrap!(args[i])
    end

    return ex
end

wrap!(ex::Number) = Expr(:call, :f, ex)
wrap!(ex) = ex

ext, x, y = copy(ex), 2, 3
@test wrap!(ex) == :(x*x + f(2)*y*x + y*y)
eval(ext)
eval(ex)
```

```@raw html
</p></details>
```

This kind of manipulation is at the core of some pkgs, such as [`IntervalArithmetics.jl`](https://github.com/JuliaIntervals/IntervalArithmetic.jl) where every number is replaced with a narrow interval in order to find some bounds on the result of a computation.

## Resources
!!! note "Where to find source code?"
    As most of Julia is written in Julia itself it is sometimes helpful to look inside for some details or inspiration. The code of `Base` and stdlib pkgs is located just next to Julia's installation in the `./share/julia` subdirectory
    ```bash
    ./julia-1.6.2/
        ├── bin
        ├── etc
        │   └── julia
        ├── include
        │   └── julia
        │       └── uv
        ├── lib
        │   └── julia
        ├── libexec
        └── share
            ├── appdata
            ├── applications
            ├── doc
            │   └── julia       # offline documentation (https://docs.julialang.org/en/v1/)
            └── julia
                ├── base        # base library
                ├── stdlib      # standard library
                └── test
    ```
    Other packages installed through Pkg interface are located in the `.julia/` directory which is located in your `$HOMEDIR`, i.e. `/home/$(user)/.julia/` on Unix based systems and `/Users/$(user)/.julia/` on Windows.
    ```bash
    ~/.julia/
        ├── artifacts
        ├── compiled
        ├── config          # startup.jl lives here
        ├── environments
        ├── logs
        ├── packages        # packages are here
        └── registries
    ```
    If you are using VSCode, the paths visible in the REPL can be clicked through to he actual source code. Moreover in that environment the documentation is usually available upon hovering over code.