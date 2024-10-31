# [Lab 06: Code introspection and metaprogramming](@id introspection_lab)
In this lab we are first going to inspect some tooling to help you understand what Julia does under the hood such as:
- looking at the code at different levels
- understanding what method is being called
- showing different levels of code optimization

Secondly we will start playing with the metaprogramming side of Julia, mainly covering:
- how to view abstract syntax tree (AST) of Julia code
- how to manipulate AST

These topics will be extended in the next lecture/lab, where we are going use metaprogramming to manipulate code with macros.

We will be again a little getting ahead of ourselves as we are going to use quite a few macros, which will be properly explained in the next lecture as well, however for now the important thing to know is that a macro is just a special function, that accepts as an argument Julia code, which it can modify.

## Quick reminder of introspection tooling
Let's start with the topic of code inspection, e.g. we may ask the following: What happens when Julia evaluates `[i for i in 1:10]`?
#### parsing 
```@repl lab06_intro
using InteractiveUtils #hide
:([i for i in 1:10]) |> dump
```
#### lowering
```@repl lab06_intro
Meta.@lower debuginfo=:none [i for i in 1:10]
```
#### typing
```@repl lab06_intro
f() = [i for i in 1:10]
@code_typed debuginfo=:none f()
```
#### LLVM code generation
```@repl lab06_intro
@code_llvm debuginfo=:none f()
```
#### native code generation
```@repl lab06_intro
@code_native debuginfo=:none f()
```

Let's see how these tools can help us understand some of Julia's internals on examples from previous labs and lectures.

### Understanding runtime dispatch and type instabilities
We will start with a question: Can we spot internally some difference between type stable/unstable code?

!!! warning "Exercise"
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
    For now do not try to understand the details, but focus on the overall differences such as length of the code.

    !!! info "Redirecting `stdout`"
        If the output of the method introspection tools is too long you can use a general way of redirecting standard output `stdout` to a file
        ```julia
        open("./llvm_fun.ll", "w") do file
            original_stdout = stdout
            redirect_stdout(file)
            @code_llvm debuginfo=:none fun()
            redirect_stdout(original_stdout)
        end
        ```
        In case of `@code_llvm` and `@code_native` there are special options, that allow this out of the box, see help `?` for underlying `code_llvm` and `code_native`. If you don't mind adding dependencies there is also the `@capture_out` from [`Suppressor.jl`](https://github.com/JuliaIO/Suppressor.jl)


### Loop unrolling
In some cases the compiler uses loop unrolling[^1] optimization to speed up loops at the expense of binary size. The result of such optimization is removal of the loop control instructions and rewriting the loop into a repeated sequence of independent statements.

[^1]: [https://en.wikipedia.org/wiki/Loop_unrolling](https://en.wikipedia.org/wiki/Loop\_unrolling)

!!! warning "Exercise"
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


### Recursion inlining depth
Inlining[^2] is another compiler optimization that allows us to speed up the code by avoiding function calls. Where applicable compiler can replace `f(args)` directly with the function body of `f`, thus removing the need to modify stack to transfer the control flow to a different place. This is yet another optimization that may improve speed at the expense of binary size.

[^2]: [https://en.wikipedia.org/wiki/Inline_expansion](https://en.wikipedia.org/wiki/Inline\_expansion)

!!! warning "Exercise"
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

    !!! info "Splatting/slurping operator `...`"
        The operator `...` serves two purposes inside function calls [^3][^4]:
        - combines multiple arguments into one
        ```@repl lab06_splat
        function printargs(args...)
            println(typeof(args))
            for (i, arg) in enumerate(args)
                println("Arg #$i = $arg")
            end
        end
        printargs(1, 2, 3)
        ```
        - splits one argument into many different arguments
        ```@repl lab06_splat
        function threeargs(a, b, c)
            println("a = $a::$(typeof(a))")
            println("b = $b::$(typeof(b))")
            println("c = $c::$(typeof(c))")
        end
        threeargs([1,2,3]...) # or with a variable threeargs(x...)
        ```

        [^3]: [https://docs.julialang.org/en/v1/manual/faq/#What-does-the-...-operator-do?](https://docs.julialang.org/en/v1/manual/faq/#What-does-the-...-operator-do?)
        [^4]: [https://docs.julialang.org/en/v1/manual/functions/#Varargs-Functions](https://docs.julialang.org/en/v1/manual/functions/#Varargs-Functions)

    **HINTS**:
    - define two methods `_polynomial!(ac, x, a...)` and `_polynomial!(ac, x, a)` for the case of ≥2 coefficients and the last coefficient
    - use splatting together with range indexing `a[1:end-1]...`
    - the correctness can be checked using the built-in `evalpoly`
    - recall that these kind of optimization are possible just around the type inference stage
    - use container of known length to store the coefficients

## AST manipulation: The first steps to metaprogramming
Julia is so called homoiconic language, as it allows the language to reason about its code. This capability is inspired by years of development in other languages such as Lisp, Clojure or Prolog.

There are two easy ways to extract/construct the code structure [^5]
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

!!! warning "Exercise"
    Before doing anything more fancy let's start with some simple manipulation of ASTs.
    - Define a variable `code` to be as the result of parsing the string `"j = i^2"`. 
    - Copy code into a variable `code2`. Modify this to replace the power `2` with a power `3`. Make sure that the original code variable is not also modified. 
    - Copy `code2` to a variable `code3`. Replace `i` with `i + 1` in `code3`.
    - Define a variable `i` with the value `4`. Evaluate the different code expressions using the `eval` function and check the value of the variable `j`.

Following up on the more general substitution of variables in an expression from the lecture, let's see how the situation becomes more complicated, when we are dealing with strings instead of a parsed AST.

!!! warning "Exercise"
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
    write a different function `sreplace_i(s)`, which does the same thing but instead of a parsed expression (AST) it manipulates a string, such as
    ```@repl lab06_meta
    s = string(ex)
    ```
    **HINTS**:
    - Use `Meta.parse` in combination with `replace_i` **ONLY** for checking of correctness.
    - You can use the `replace` function in combination with regular expressions.
    - Think of some corner cases, that the method may not handle properly.

If the exercises so far did not feel very useful let's focus on one, that is similar to a part of the [`IntervalArithmetics.jl`](https://github.com/JuliaIntervals/IntervalArithmetic.jl) pkg.

!!! warning "Exercise"
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

This kind of manipulation is at the core of some pkgs, such as aforementioned [`IntervalArithmetics.jl`](https://github.com/JuliaIntervals/IntervalArithmetic.jl) where every number is replaced with a narrow interval in order to find some bounds on the result of a computation.

---
[^5]: Once you understand the recursive structure of expressions, the AST can be constructed manually like any other type.

## Resources
- Julia's manual on [metaprogramming](https://docs.julialang.org/en/v1/manual/metaprogramming/)
- David P. Sanders' [workshop @ JuliaCon 2021](https://www.youtube.com/watch?v=2QLhw6LVaq0) 
- Steven Johnson's [keynote talk @ JuliaCon 2019](https://www.youtube.com/watch?v=mSgXWpvQEHE)
- Andy Ferris's [workshop @ JuliaCon 2018](https://www.youtube.com/watch?v=SeqAQHKLNj4)
- [From Macros to DSL](https://github.com/johnmyleswhite/julia_tutorials) by John Myles White 
- Notes on [JuliaCompilerPlugin](https://hackmd.io/bVhb97Q4QTWeBQw8Rq4IFw?both#Julia-Compiler-Plugin-Project)
