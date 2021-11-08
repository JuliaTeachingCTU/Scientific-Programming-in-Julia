# [Macros](@id macro_lecture)
What is macro?
In its essence, macro is a function, which 
1. takes as an input an expression (parsed input)
2. modify the expressions in argument
3. insert the modified expression at the same place as the one that is parsed.

Macros are necessary because they execute when code is parsed, therefore, macros allow the programmer to generate and include fragments of customized code before the full program is run. To illustrate the difference, consider the following example:

One of the very conveninet ways to write macros is to write functions modifying the `Expr`ession and then call that function in macro as 
```julia
replace_sin(x::Symbol) = x == :sin ? :cos : x
replace_sin(e::Expr) = Expr(e.head, map(replace_sin, e.args)...)
replace_sin(u) = u

macro replace_sin(ex)
	replace_sin(esc(ex))
end

@replace_sin(cosp1(x) = 1 + sin(x))
cosp1(1) == 1 + cos(1)
```
notice the following
- the definition of the macro is similar to the definition of the function with the exception that instead of the keyword `function` we use keyword `macro`
- when calling the macro, we signal to the compiler our intention by prepending the name of the macro with `@`. 
- the macro receives the expression(s) as the argument instead of the evaluated argument and also returns an expression that is placed on the position where the macro has been called
- when you are invoking the macro, you should be aware that the code you are entering can be arbitrarily modified and you can receive something completely different. This meanst that `@` should also serve as a warning that you are leaving Julia's syntax. In practice, it make sense to make things akin to how they are done in Julia or to write Domain Specific Language with syntax familiar in that domain.

We have mentioned above that macros are indispensible in the sense they intercept the code generation after parsing. You might object that I can achieve the above using the following combination of `Meta.parse` and `eval`
```julia
ex = Meta.parse("cosp1(x) = 1 + sin(x)")
ex = replace_sin(ex)
eval(ex)
```
in the following we cannot do the same trick
```julia
function cosp2(x)
	@replace_sin 2 + sin(x)
end
cosp2(1) ≈ (2 + cos(1))
```

```julia
function parse_eval_cosp2(x)
	ex = Meta.parse("2 + sin(x)")
	ex = replace_sin(ex)
	eval(ex)
end

julia> @code_lowered parse_eval_cosp2(1)
CodeInfo(
1 ─ %1 = Base.getproperty(Main.Meta, :parse)
│        ex = (%1)("2 + sin(x)")
│        ex = Main.replace_sin(ex)
│   %4 = Main.eval(ex)
└──      return %4
)
```

!!! info 
	### Scope of eval
	`eval` function is always evaluated in the global scope of the `Module` in which the macro is called (note that there is that by default you operate in the `Main` module). Moreover, `eval` takes effect **after** the function has been has been executed. This can be demonstrated as 
	```julia
	add1(x) = x + 1
	function redefine_add(x)
		eval(:(add1(x) = x - 1))
		add1(x)
	end
	julia> redefine_add(1)
	2

	julia> redefine_add(1)
	0
	```


`@macroexpand` can allow use to observe, how the macro will be expanded. We can use it for example 
```julia
@macroexpand @replace_sin(sinp1(x) = 1 + sin(x))
```

## What goes under the hood?
Let's consider what the compiler is doing in this call
```julia
function cosp2(x)
	@replace_sin 2 + sin(x)
end
```

First, Julia parses the code into the AST as
```julia
ex = Meta.parse("""
   function cosp2(x)
	   @replace_sin 2 + sin(x)
end
""") |> Base.remove_linenums!
dump(ex)
```
We observe that there is a macrocall in the AST, which means that Julia will expand the macro and put it in place
```julia
ex.args[2].args[1].head 	# the location of the macrocall
ex.args[2].args[1].args[1]  # which macro to call
ex.args[2].args[1].args[2]  # line number
ex.args[2].args[1].args[3]	# on which expression
```
let's run the `replace_sin` and insert it back
```julia
ex.args[2].args[1] = replace_sin(ex.args[2].args[1].args[3])
ex |> dump
```
now, `ex` contains the expanded macro and we can see that it correctly defines the function
```julia
eval(ex)
```
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

Observe that macro dispatch is based on the types of AST that are handed to the macro, not the types that the AST evaluates to at runtime.

## Notes on quotation
In the previous lecture we have seen that we can *quote a block of code*, which tells the compiler to treat the input as an data and parse it. We have talked about three ways of quoting code.
1.  `:(quoted code)`
2. Meta.parse(input_string)
3. `quote ... end`
The truth is that Julia does not do full quotation, but a *quasiquotation* is it allows you to **interpolate** expressions inside the quoted code using `$` symbol similar to the string. This is handy, as sometimes, when we want to insert into the quoted code an result of some computation / preprocessing.
Observe the following difference in returned code
```julia
a = 5
:(x = a)
:(x = $(a))
let y = :x
    :(1 + y), :(1 + $y)
end
```
In contrast to the behavior of `:()` (or `quote ... end`, true quotation would not perform interpolation where unary `$` occurs. Instead, we would capture the syntax that describes interpolation and produce something like the following:
```julia
(
    :(1 + x),                        # Quasiquotation
    Expr(:call, :+, 1, Expr(:$, :x)), # True quotation
)
```

When we need true quoting, i.e. we need something to stay quoted, we can use `QuoteNode` as
```julia
macro true_quote(e)
    QuoteNode(e)
end
let y = :x
    (
        @true_quote(1 + $y),
        :(1 + $y),
    )
end
```
At first glance, `QuoteNode` wrapper seems to be useless. But `QuoteNode` has clear value when it's used inside a macro to indicate that something should stay quoted even after the macro finishes executing. Also notice that the expression received by macro was quoted, not quasiquoted, since in the latter case `$y` would be replaced. We can demonstate it using the `@showarg` macro introduced earlier, as
```julia
@showarg(1 + $x)
```
The error is raised after the macro was evaluated and the output has been inserted to parsed AST.

Macros do not know about runtime values, they only know about syntax trees. When a macro receives an expression with a $x in it, it can't interpolate the value of x into the syntax tree because it reads the syntax tree before x ever has a value! So the interpolation syntax in macros is not given any actual meaning in julia.

Instead, when a macro is given an expression with $ in it, it assumes you're going to give your own meaning to $x. In the case of BenchmarkTools.jl they return code that has to wait until runtime to receive the value of x and then splice that value into an expression which is evaluated and benchmarked. Nowhere in the actual body of the macro do they have access to the value of x though.


!!! info 
	### Why `$` for interpolation?
	The `$` string for interpolation was used as it identifies the interpolation inside the string and inside the command. For example
	```julia
	a = 5
	s = "a = $(5)"
	typoef(s)
	println(s)
	filename = "/tmp/test_of_interpolation"
	run(`touch $(filename)`)
	```

## Macro hygiene
Macro hygiene is a term coined in 1986 and it says that the evaluation of the macro should not have an effect on the surrounding call. By default, all macros in Julia are hygienic which means that variables introduced in the macro are `gensym`ed to have unique names and function points to global functions.

Let's demonstrate it on our own version of an macro `@elapsed` which will return the time that was needed to evaluate the block of code.
```julia
macro tooclean_elapsed(ex)
	quote
		tstart = time()
		$(ex)
		time() - tstart
	end
end

fib(n) = n <= 1 ? n : fib(n-1) + fib(n - 2)
let 
	t = @tooclean_elapsed r = fib(10)
	println("the evaluation of fib took ", t, "s and result is ", r)
end
```
We see that variable `r` has not been assigned during the evaluation of macro. We have also used `let` block in orders not to define any variables in the global scope.
Why is that?
Let's observe how the macro was expanded
```julia
julia> Base.remove_linenums!(@macroexpand @tooclean_elapsed r = fib(10))
quote
    var"#12#tstart" = Main.time()
    var"#13#r" = Main.fib(10)
    Main.time() - var"#12#tstart"
end
```
We see that `tstart` in the macro definition was replaced by `var"#12#tstart"`, which is a name generated by Julia's gensym to prevent conflict. The same happens to `r`, which was replaced by `var"#13#r"`. Notice that in the case of `tstart`, we actually want to replace `tstart` with a unique name, such that if we by a bad luck define `tstart` in our code, it would not be affected, as we can see in this example.
```julia
let 
	tstart = "should not change the value and type "
	t = @tooclean_elapsed r = fib(10)
	println(tstart, "  ", typeof(tstart))
end
```
But in the second case, we would actually very much like the variable `r` to retain its name, such that we can accesss the results (and also, `ex` can access and change other local variables). Julia offer a way to `escape` from the hygienic mode, which means that the variables will be used and passed as-is. Notice the effect if we escape jus the expression `ex` 

```julia
macro justright_elapsed(ex)
	quote
		tstart = time()
		$(esc(ex))
		time() - tstart
	end
end

let 
	tstart = "should not change the value and type "
	t = @justright_elapsed r = fib(10)
	println("the evaluation of fib took ", t, "s and result is ", r)
	println(tstart, "  ", typeof(tstart))
end
```
which now works as intended. We can inspect the output again using `@macroexpand`
```julia
julia> Base.remove_linenums!(@macroexpand @justright_elapsed r = fib(10))
quote
    var"#19#tstart" = Main.time()
    r = fib(10)
    Main.time() - var"#19#tstart"
end
```
and compare it to `Base.remove_linenums!(@macroexpand @justright_elapsed r = fib(10))`. We see that the experssion `ex` has its symbols intact.
To use the escaping / hygience correctly, you need to have a good understanding how the macro evaluation works and what is needed. Let's now try the third version of the macro, where we escape everything as
```julia
macro toodirty_elapsed(ex)
	ex = quote
		tstart = time()
		$(ex)
		time() - tstart
	end
	esc(ex)
end

let 
	tstart = "should not change the value and type "
	t = @toodirty_elapsed r = fib(10)
	println("the evaluation of fib took ", t, "s and result is ", r)
	println(tstart, "  ", typeof(tstart))
end
```

```julia
julia> Base.remove_linenums!(@macroexpand @toodirty_elapsed r = fib(10))
quote
    tstart = time()
    r = fib(10)
    time() - tstart
end
```


!!! info 
	### gensym

	`gensym([tag])` Generates a symbol which will not conflict with other variable names.
	```julia
	julia> gensym("hello")
	Symbol("##hello#257")
```



but if we look, how the function is expanded, we see that it is not as we have expected
```julia

macro replace_sin(ex)
	replace_sin(esc(ex))
end

function cosp2(x)
	@replace_sin 2 + sin(x)
end

julia> @code_lowered(cosp2(1.0))
CodeInfo(
1 ─ %1 = Main.cos(Main.x)
│   %2 = 2 + %1
└──      return %2
)
```
why is that

## DSL
## non-standard string literals
```
macro r_str(p)
    Regex(p)
end
```
## Write @exfiltrate macro
`Base.@locals`

## sources
Great discussion on evaluation of macros
https://discourse.julialang.org/t/interpolation-in-macro-calls/25530