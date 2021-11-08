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
Macro hygiene is a term coined in 1986. The problem it addresses is following: if you're automatically generating code, it's possible that you will introduce variable names in your generated code that will clash with existing variable names in the scope in which a macro is called. These clashes might cause your generated code to read from or write to variables that you should not interacting with. A macro is hygienic when it does not interact with existing variables, which means that when macro is evaluated, it should not have any effect on the surrounding code. 

By default, all macros in Julia are hygienic which means that variables introduced in the macro have automatically generated names, where Julia ensures they will not collide with user's variable. These variables are created by `gensym` function / macro. 

!!! info 
	### gensym

	`gensym([tag])` Generates a symbol which will not conflict with other variable names.
	```julia
	julia> gensym("hello")
	Symbol("##hello#257")
```

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
We see that `tstart` in the macro definition was replaced by `var"#12#tstart"`, which is a name generated by Julia's gensym to prevent conflict. The same happens to `r`, which was replaced by `var"#13#r"`. This names are the result of Julia's hygiene-enforcing pass, which is intended to prevent us from overwriting existing variables during macro expansion. This pass usually makes our macros safer, but it is also a source of confusion because it introduces a gap between the expressions we generate and the expressions that end up in the resulting source code. Notice that in the case of `tstart`, we actually wanted to replace `tstart` with a unique name, such that if we by a bad luck define `tstart` in our code, it would not be affected, as we can see in this example.
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
Using `@macroexpand` we observe that `@toodirty_elapsed` does not have any trace of hygiene.
```julia
julia> Base.remove_linenums!(@macroexpand @toodirty_elapsed r = fib(10))
quote
    tstart = time()
    r = fib(10)
    time() - tstart
end
```
From the above we can also see that hygiene-pass occurs after the macro has been applied but before the code is lowered. `esc` is inserted to AST as a special node `Expr(:escape,...),` which can be seen from the follows.
```julia
julia> esc(:x)
:($(Expr(:escape, :x)))
```
The definition in `essentials.jl:480` is pretty simple as `esc(@nospecialize(e)) = Expr(:escape, e)`.

With that in mind, we can now understand our original example with `@replace_sin`. Recall that we have defined it as 
```julia
macro replace_sin(ex)
	replace_sin(esc(ex))
end
```
where the escaping `replace_sin(esc(ex))` in communicates to compiler that `ex` should be used as without hygienating the `ex`.  Indeed, if we lower it
```julia
function cosp2(x)
	@replace_sin 2 + sin(x)
end

julia> @code_lowered(cosp2(1.0))
CodeInfo(
1 ─ %1 = Main.cos(x)
│   %2 = 2 + %1
└──      return %2
)
```
we see it works as intended. Whereas if we use hygienic version
```julia
macro hygienic_replace_sin(ex)
	replace_sin(ex)
end

function hcosp2(x)
	@hygienic_replace_sin 2 + sin(x)
end

julia> @code_lowered(hcosp2(1.0))
CodeInfo(
1 ─ %1 = Main.cos(Main.x)
│   %2 = 2 + %1
└──      return %2
)
```


### Can I do the hygiene by myself?
Yes, it is by some considered to be much simpler (and safer) then to understand, how macro hygiene works.
```julia
macro manual_elapsed(ex)
    x = gensym()
    esc(quote
    		$(x) = time()
        	$(ex)
        	time() - $(x)
        end
    )
end

let 
	t = @manual_elapsed r = fib(10)
	println("the evaluation of fib took ", t, "s and result is ", r)
end

```

## How macros compose?
```julia
macro m1(ex)
	println("m1: ")
	dump(ex)
	ex
end

macro m2(ex)
	println("m2: ")
	dump(ex)
	esc(ex)
end

@m1 @m2 1 + sin(1)
```
which means that macros are expanded in the order from the outer most to inner most, which is exactly the other way around than functions.
```
@macroexpand @m1 @m2 1 + sin(1)
```
also notice that the escaping is only partial (running `@macroexpand @m2 @m1 1 + sin(1)` would not change the results).

<!-- Why are variables on right-hand side replaced by access to global variables? -->
## Write @exfiltrate macro
`Base.@locals`


## Domain Specifis Languages (DSL)
Macros are convenient for writing domain specific languages, which are languages designed for specific domain. This allows them to simplify notation and / or make the notation familiar for people working in the field. For example in `Turing.jl`, the model 
of coinflips can be specified as 
```
@model function coinflip(y)

    # Our prior belief about the probability of heads in a coin.
    p ~ Beta(1, 1)

    # The number of observations.
    N = length(y)
    for n in 1:N
        # Heads or tails of a coin are drawn from a Bernoulli distribution.
        y[n] ~ Bernoulli(p)
    end
end;
```
which resembles, but not copy Julia's syntax due to the use of `~`. A similar DSLs can be seen in `ModelingToolkit.jl` for differential equations, in `Soss.jl` again for expressing probability problems, in `Metatheory.jl` / `SymbolicUtils.jl` for defining rules on elements of algebras, or `JuMP.jl` for specific mathematical programs.

One of the reasons for popularity of DSLs is that macro system is very helpful in their implementation, but it also contraints the DSL, as it has to be parseable by Julia's parser. This is a tremendous helps, because one does not have to care about how to parse numbers, strings, parenthesess, functions, etc. 

Let's jump into the first example adapted from (John Myles White's howto)[https://github.com/johnmyleswhite/julia_tutorials/blob/master/From%20Macros%20to%20DSLs%20in%20Julia%20-%20Part%202%20-%20DSLs.ipynb].
We would like to write a macro, which allows us to define graph in `LightGraphs.jl` just by defining edges.
```julia
@graph begin 
	1 -> 2
	2 -> 3
	3 -> 1
end
``` 
The above should expand to
```julia
using LightGraphs
g = DiGraph(3)
add_edge!(g, 1,2)
add_edge!(g, 2,3)
add_edge!(g, 3,1)
d
```
Let's start with easy and observe, how 
```julia
ex = Meta.parse("""
begin 
	1 -> 2
	2 -> 3
	3 -> 1
end
""")
ex = Base.remove_linenums!(ex)
```
is parsed to 
```julia
quote
    1->begin
            2
        end
    2->begin
            3
        end
    3->begin
            1
        end
end
```
We see that 
- the sequence of statements is parsed to `block` (we know that from last lecture).
- `->` is parsed to `->`, i.e. `ex.args[1].head == :->` with parameters being the first vertex `ex.args[1].args[1] == 1` and the second vertex is quoted to `ex.args[1].args[2].head == :block`. 

The main job will be done in the function `parse_edge`, which will parse one edge. It will check that the node defines edge (otherwise, it will return nothing, which will be filtered out)
```julia
function parse_edge(ex)
	#checking the syntax
	!hasproperty(ex, :head) && return(nothing)
	!hasproperty(ex, :args) && return(nothing)
	ex.head != :-> && return(nothing)
	length(ex.args) != 2 && return(nothing)
	!hasproperty(ex.args[2], :head) && return(nothing)
	ex.args[2].head != :block && length(ex.args[2].args) == 1 && return(nothing)

	#ready to go
	src = ex.args[1]
	@assert src isa Integer
	dst = ex.args[2].args[1]
	@assert dst isa Integer
	:(add_edge!(g, $(src), $(dst)))
end

function parse_graph(ex)
	@assert ex.head == :block
	ex = Base.remove_linenums!(ex)
	edges = filter(!isnothing, parse_edge.(ex.args))
	n = maximum(e -> maximum(e.args[3:4]), edges)
	quote
       g = LightGraphs.DiGraph($(n))
       $(edges...)
       g
   end
end
```
Once we have the first version, let's make everything hygienic

```julia
function parse_edge(g, ex::Expr)
	#checking the syntax
	ex.head != :-> && return(nothing)
	length(ex.args) != 2 && return(nothing)
	!hasproperty(ex.args[2], :head) && return(nothing)
	ex.args[2].head != :block && length(ex.args[2].args) == 1 && return(nothing)

	#ready to go
	src = ex.args[1]
	@assert src isa Integer
	dst = ex.args[2].args[1]
	@assert dst isa Integer
	:(add_edge!($(g), $(src), $(dst)))
end
parse_edge(g, ex) = nothing

function parse_graph(ex)
	@assert ex.head == :block
	g = gensym(:graph)
	ex = Base.remove_linenums!(ex)
	edges = filter(!isnothing, parse_edge.(g, ex.args))
	n = maximum(e -> maximum(e.args[3:4]), edges)
	quote
       $(g) = LightGraphs.DiGraph($(n))
       $(edges...)
       $(g)
   end
end
```
and we are ready to go
```julia
macro graph(ex)
	parse_graph(ex)
end

@graph begin
	1 -> 2
	2 -> 3
	3 -> 1
end
```
and we can check the output with `@macroexpand`.
```julia
julia> @macroexpand @graph begin
               1 -> 2
               2 -> 3
               3 -> 1
       end
quote
    #= REPL[173]:8 =#
    var"#27###graph#273" = (Main.LightGraphs).DiGraph(3)
    #= REPL[173]:9 =#
    Main.add_edge!(var"#27###graph#273", 1, 2)
    Main.add_edge!(var"#27###graph#273", 2, 3)
    Main.add_edge!(var"#27###graph#273", 3, 1)
    #= REPL[173]:10 =#
    var"#27###graph#273"
end
```

## Symbolic differentiation of scalars

## non-standard string literals
```
macro r_str(p)
    Regex(p)
end
```
## sources
Great discussion on evaluation of macros
https://discourse.julialang.org/t/interpolation-in-macro-calls/25530