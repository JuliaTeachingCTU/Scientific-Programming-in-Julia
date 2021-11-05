# [Macros](@id macro_lecture)
What is macro?
In its essence, macro is a function, which 
1. takes as an input an expression (or more expression)
2. modify the expressions in argument
3. evaluate the modified expression exactly once during the compile time. The expression returned from the macro is inserted at the position of the macro and evaluated. After the macro is applied there is no trace of macro being called (from the syntactic point of view).

Thus, the macro can be thus viewed as a convenience for performing
```
ex = Meta.parse(some_source_code)
eval(replace_sin(ex))
```

We can instantiate the aboce with the following example
```julia
replace_sin(x::Symbol) = x == :sin ? :cos : x
replace_sin(e::Expr) = Expr(e.head, map(replace_sin, e.args)...)
replace_sin(u) = u
some_source_code = "sinp1(x) = 1 + sin(x)"

ex = Meta.parse(some_source_code)
eval(replace_sin(ex))
sinp1(1) == 1 + cos(1)
```
which can be equally written by defining macro using keyword `macro` 
```
macro replace_sin(ex)
	o = replace_sin(ex)
end
@replace_sin(sinp1(x) = 1 + sin(x))
sinp1(1) == 1 + cos(1)
```
notice the following
- the definition of the macro is similar to the definition of the function with the exception that instead of the keyword `function` we use keyword `macro`
- when calling the macro, we signal to the compiler our intention by prepending the name of the macro with `@`. 
- the macro receives the expression(s) as the argument instead of the evaluated argument
- when you are invoking the macro, you should be aware that the code you are entering can be arbitrarily modified and you can receive something completely different (this of course does not make sense from the functional perspective).

## Calling macros
Macros can be called without parentheses
```julia
macro showarg(ex)
	println("single argument version")
	@show ex
	ex
end
@showarg 1 + 1
@showarg(1 + 1)
```
but they use the very same multiple dispatch as functions
```julia
macro showarg(x1, x2::Symbol)
	println("two argument version, second is Symbol")
	x1
end
macro showarg(x1, x2::Expr)
	println("two argument version, second is Symbol")
	x1
end
@showarg(1 + 1, x)
@showarg(1 + 1, 1 + 3)
@showarg 1 + 1, 1 + 3
@showarg 1 + 1  1 + 3
```
(the `@showarg(1 + 1, :x) ` raises an error, since `:(:x)` is of Type `QuoteNode`).

<!-- We can observe the AST corresponding to the macro call using
```julia
eval(Expr(:macrocall, Symbol("@showarg"), :(1 + 1)))
Meta.parse("@showarg 1 + 1")
``` -->

`@macroexpand` can allow use to observe, how the macro will be expanded. We can use it for example 
```julia
@macroexpand @replace_sin(sinp1(x) = 1 + sin(x))
```


## Basics
## non-standard string literals
## Macro hygiene
## DSL
## Write @exfiltrate macro
`Base.@locals`