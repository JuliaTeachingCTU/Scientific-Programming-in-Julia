# [Lab 06: Code introspection and metaprogramming](@id introspection_lab)
In this lab we are first going to inspect some tooling to help you understand what Julia does under the hood such as:
- looking at the code at different levels
- understanding what method is being called
- showing different levels of code optimization

Secondly we will start playing with the metaprogramming side of Julia, mainly covering:
- how to view abstract syntax tree (AST) of Julia code
- how to manipulate AST

These topics will be extended in the next [lecture](@ref macro_lecture)/[lab](@ref macro_lab), where we are going use metaprogramming to manipulate code with macros.

We will be again a little getting ahead of ourselves as we are going to use quite a few macros, which will be properly explained in the next lecture as well, however for now the important thing to know is that a macro is just a special function, that accepts as an argument Julia code, which it can modify.

## Quick reminder of introspection tooling
Let's start with the topic of code inspection, e.g. we may ask the following: What happens when Julia evaluates `[i for i in 1:10]`?
- parsing 
```@repl lab06_intro
using InteractiveUtils #hide
:([i for i in 1:10]) |> dump
```
- lowering
```@repl lab06_intro
Meta.@lower [i for i in 1:10]
```
- typing
```@repl lab06_intro
f() = [i for i in 1:10]
@code_typed f()
```
- LLVM code generation
```@repl lab06_intro
@code_llvm f()
```
- native code generation
```@repl lab06_intro
@code_native f()
```

Let's see how these tools can help us understand some of Julia's internals on examples from previous labs and lectures.

### Understanding the runtime dispatch and type instabilities
We will start with a question: Can we spot internally some difference between type stable/unstable code?

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Inspect the following two functions using `@code_lowered`, `@code_typed`, `@code_llvm` and `@code_native`.
```@example lab06_intro
x = rand(10^5)
function explicit_len(x)
    length(x)
end

function implicit_len()
    length(x)
end
nothing #hide
```

!!! info "Redirecting `stdout`"
    If the output of the method introspection tools is too long you can use a general way of redirecting standard output `stdout` to a file
    ```julia
    open("./llvm_fun.ll", "w") do file
        original_stdout = stdout
        redirect_stdout(file)
        @code_llvm fun()
        redirect_stdout(original_stdout)
    end
    ```
    In case of `@code_llvm` and `@code_native` there are special options, that allow this out of the box, see help `?` for underlying `code_llvm` and `code_native`. If you don't mind adding dependencies there is also the `@capture_out` from [`Suppressor.jl`](https://github.com/JuliaIO/Suppressor.jl)

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
@code_warntype explicit_sum(x)
@code_warntype implicit_sum()

@code_typed explicit_sum(x)
@code_typed implicit_sum()

@code_llvm explicit_sum(x)
@code_llvm implicit_sum()

@code_native explicit_sum(x)
@code_native implicit_sum()
```

In this case we see that the generated code for such a simple operation is much longer in the type unstable case resulting in longer run times. However in the next example we will see that having longer code is not always a bad thing.
```@raw html
</p></details>
```

### Loop unrolling
In some cases the compiler uses loop unrolling[^1] optimization to speed up loops at the expense of binary size. The result of such optimization is removal of the loop control instructions and rewriting the loop into a repeated sequence of independent statements.

[^1]: [https://en.wikipedia.org/wiki/Loop_unrolling](https://en.wikipedia.org/wiki/Loop\_unrolling)

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Inspect under what conditions does the compiler unroll the for loop in the `polynomial` function from the last [lab](@ref horner).
```@example lab06_intro
function polynomial(a, x)
    accumulator = a[end] * one(x)
    for i in length(a)-1:-1:1
        accumulator = accumulator * x + a[i]
    end
    accumulator  
end
nothing #hide
```

Compare the speed of execution with and without loop unrolling.

**HINTS**:
- these kind of optimization are lower level than intermediate language
- loop unrolling is possible when compiler knows the length of the input

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```@example lab06_intro
using Test #hide
using BenchmarkTools
a = Tuple(ones(20)) # tuple has known size
ac = collect(a)
x = 2.0

@code_lowered polynomial(a,x)       # cannot be seen here as optimizations are not applied
@code_typed polynomial(a,x)         # loop unrolling is not part of type inference optimization
nothing #hide
```

```@repl lab06_intro
@code_llvm polynomial(a,x)
@code_llvm polynomial(ac,x)
```

More than 2x speedup
```@repl lab06_intro
@btime polynomial($a,$x)
@btime polynomial($ac,$x)
```

```@raw html
</p></details>
```

### Recursion inlining depth
Inlining[^2] is another compiler optimization that allows us to speed up the code by avoiding function calls. Where applicable compiler can replace `f(args)` directly with the function body of `f`, thus removing the need to modify stack to transfer the control flow to a different place. This is yet another optimization that may improve speed at the expense of binary size.

[^2]: [https://en.wikipedia.org/wiki/Inline_expansion](https://en.wikipedia.org/wiki/Inline\_expansion)

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Rewrite the `polynomial` function from the last [lab](@ref horner) using recursion and find the length of the coefficients, at which inlining of the recursive calls stops occurring.

```julia
function polynomial(a, x)
    accumulator = a[end] * one(x)
    for i in length(a)-1:-1:1
        accumulator = accumulator * x + a[i]
    end
    accumulator  
end
```

**HINTS**:
- define two methods `_polynomial(x, a...)` and `_polynomial(x, a)`
- recall that these kind of optimization are run just after type inference
- use container of known length to store the coefficients

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@example lab06_intro
_polynomial!(ac, x, a...) = _polynomial!(x * ac + a[end], x, a[1:end-1]...)
_polynomial!(ac, x, a) = x * ac + a
polynomial(a, x) = _polynomial!(a[end] * one(x), x, a[1:end-1]...)

# the coefficients have to be a tuple
a = Tuple(ones(Int, 21)) # everything less than 22 gets inlined
x = 2
polynomial(a,x) == evalpoly(x,a) # compare with built-in function

@code_lowered polynomial(a,x) # cannot be seen here as optimizations are not applied
@code_llvm polynomial(a,x)    # seen here too, but code_typed is a better option
nothing #hide
```

```@repl lab06_intro
@code_typed polynomial(a,x)
```

```@raw html
</p></details>
```

## AST manipulation: The first steps to metaprogramming
Julia is so called homoiconic language, as it allows the language to reason about its code. This capability is inspired by years of development in other languages such as Lisp, Clojure or Prolog.

There are two easy ways to extract/construct the code structure [^3]
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
- constructing an expression using `quote ... end` or simple `:()` syntax
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
using GraphRecipes #hide
using Plots #hide
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

Following up on the more general substitution of variables in an expression from the lecture, let's see how the situation becomes more complicated, when we are dealing with strings instead of a parsed AST.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
```@example lab06_meta
using Test #hide
replace_i(s::Symbol) = s == :i ? :k : s
replace_i(e::Expr) = Expr(e.head, map(replace_i, e.args)...)
replace_i(u) = u
nothing #hide
```
Given a function `replace_i`, which replaces variables `i` for `k` in an expression like the following
```@repl lab06_meta
ex = :(i + i*i + y*i - sin(z))
@test replace_i(ex) == :(k + k*k + y*k - sin(z))
```
write function `sreplace_i(s)` which does the same thing but instead of a parsed expression (AST) it manipulates a string, such as
```@repl lab06_meta
s = string(ex)
```
Think of some corner cases, that the method may not handle properly.

**HINTS**:
- Use `Meta.parse` in combination with `replace_i` **ONLY** for checking of correctness.
- You can use the `replace` function.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
The naive solution
```@repl lab06_meta
sreplace_i(s) = replace(s, 'i' => 'k')
@test Meta.parse(sreplace_i(s)) == replace_i(Meta.parse(s))
```
does not work in this simple case, because it will replace "i" inside the `sin(z)` expression. Avoiding these corner cases would require some more involved logic (like complicated regular expressions in `replace`), therefore using the parsed AST is preferable when manipulating the code.

```@raw html
</p></details>
```

If the exercises so far did not feel very useful let's focus on one, that is similar to a part of the [`IntervalArithmetics.jl`](https://github.com/JuliaIntervals/IntervalArithmetic.jl) pkg.
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Write function `wrap!(ex::Expr)` which wraps literal values (numbers) with a call to `f()`. You can test it on the following example
```@example lab06_meta
f = x -> convert(Float64, x)
ex = :(x*x + 2*y*x + y*y)     # original expression
rex = :(x*x + f(2)*y*x + y*y) # result expression
nothing #hide
```

**HINTS**:
- use recursion and multiple dispatch
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

This kind of manipulation is at the core of some pkgs, such as aforementioned [`IntervalArithmetics.jl`](https://github.com/JuliaIntervals/IntervalArithmetic.jl) where every number is replaced with a narrow interval in order to find some bounds on the result of a computation.


[^3]: Once you understand the recursive structure of expressions, the AST can be constructed manually like any other type.
---

## Resources
- Julia's manual on [metaprogramming](https://docs.julialang.org/en/v1/manual/metaprogramming/)
- David P. Sanders' [workshop @ JuliaCon 2021](https://www.youtube.com/watch?v=2QLhw6LVaq0) 
- Steven Johnson's [keynote talk @ JuliaCon 2019](https://www.youtube.com/watch?v=mSgXWpvQEHE)
- Andy Ferris's [workshop @ JuliaCon 2018](https://www.youtube.com/watch?v=SeqAQHKLNj4)
- [From Macros to DSL](https://github.com/johnmyleswhite/julia_tutorials) by John Myles White 
- Notes on [JuliaCompilerPlugin](https://hackmd.io/bVhb97Q4QTWeBQw8Rq4IFw?both#Julia-Compiler-Plugin-Project)