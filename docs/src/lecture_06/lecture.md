# [Language introspection](@id introspection)

**What is metaprogramming?** *A high-level code that writes high-level code* by Stever Johnson.

**Why do we need metaprogramming?** 
- In general, we do not need it, as we can do whatever we need without it, but it can help us to remove a boilerplate code. 
    + As an example, consider a `@show` macro, which just prints the name of the variable (or the expression) and its evaluation. This means that instead of writing `println("2+exp(4) = ",  2+exp(4))` we can just write `@show 2+exp(4)`.
    + Another example is `@time` or `elapsed` The above is difficult to implement using normal function, since when you pass `2+exp(4)` as a function argument, it will be automatically evaluated. Therefore you need to pass it as an expression, that can be evaluated within the function.
    + It can help when implementing **encapsulation**.
- **Domain Specific Languages**


## Stages of compilation
Julia (as any modern compiler) uses several stages to convert source code to native code. Let's recap them
- parsing the source code to **abstract syntax tree** (AST)
- lowering the abstract syntax tree **static single assignment** form (SSA) [see wiki](https://en.wikipedia.org/wiki/Static_single_assignment_form)
- assigning types to variables and performing type inference on called functions
- lowering the typed code to LLVM intermediate representation (LLVM Ir)
- using LLVM compiler to produce a native code.


### Example: Fibonacci numbers
Consider for example a function computing the Fibonacci numbers[^1]
```julia
function nextfib(n)
	a, b = one(n), one(n)
	while b < n
		a, b = b, a + b
	end
	return b
end
```

[^1]: From StackOverflow [https://stackoverflow.com/questions/43453944/what-is-the-difference-between-code-native-code-typed-and-code-llvm-in-julia](https://stackoverflow.com/questions/43453944/what-is-the-difference-between-code-native-code-typed-and-code-llvm-in-julia)


#### Parsing
The first thing the compiler does is that it will parse the source code (represented as a string) to the abstract syntax tree (AST). We can inspect the results of this stage as 
```julia
julia> parsed_fib = Meta.parse(
"""
	function nextfib(n)
		a, b = one(n), one(n)
		while b < n
			a, b = b, a + b
		end
		return b
	end""")
:(function nextfib(n)
      #= none:1 =#
      #= none:2 =#
      (a, b) = (one(n), one(n))
      #= none:3 =#
      while b < n
          #= none:4 =#
          (a, b) = (b, a + b)
      end
      #= none:6 =#
      return b
  end)
```
AST is a tree representation of the source code, where the parser has already identified individual code elements function call, argument blocks, etc. The parsed code is represented by Julia objects, therefore it can be read and modified by Julia from Julia at your wish (this is what is called homo-iconicity of a language the itself being derived from Greek words *homo*- meaning "the same" and *icon* meaning "representation"). Using `TreeView`
```julia
using TreeView, TikzPictures
g = tikz_representation(walk_tree(parsed_fib))
TikzPictures.save(SVG("parsed_fib.svg"), g)
```
![parsed_fib.svg](parsed_fib.svg)

We can see that the AST is indeed a tree, with `function` being the root node (caused by us parsing a function). Each inner node represents a function call with children of the inner node being its arguments. An interesting inner node is the `Block` representing a sequence of statements, where we can also see information about lines in the source code inserted as comments. Lisp-like S-Expression can be printed using `Meta.show_sexpr(parsed_fib)`.
```julia
(:function, (:call, :nextfib, :n), (:block,
    :(#= none:1 =#),
    :(#= none:2 =#),
    (:(=), (:tuple, :a, :b), (:tuple, (:call, :one, :n), (:call, :one, :n))),
    :(#= none:3 =#),
    (:while, (:call, :<, :b, :n), (:block,
        :(#= none:4 =#),
        (:(=), (:tuple, :a, :b), (:tuple, :b, (:call, :+, :a, :b)))
      )),
    :(#= none:6 =#),
    (:return, :b)
  ))
```

#### Lowering
The next stage is **lowering**, where AST is converted to Static Single Assignment Form (SSA), in which "each variable is assigned exactly once, and every variable is defined before it is used". Loops and conditionals are transformed into gotos and labels using a single unless/goto construct (this is not exposed in user-level Julia).
```julia
julia> @code_lowered nextfib(3)
CodeInfo(
1 ─ %1 = Main.one(n)
│   %2 = Main.one(n)
│        a = %1
└──      b = %2
2 ┄ %5 = b < n
└──      goto #4 if not %5
3 ─ %7 = b
│   %8 = a + b
│        a = %7
│        b = %8
└──      goto #2
4 ─      return b
)
```
or alternatively `lowered_fib = Meta.lower(@__MODULE__, parsed_fib)`. 

We can see that 
- compiler has introduced a lot of variables 
- `while` (and `for`) loops has been replaced by a `goto` prepended by conditional statements

For inserted debugging information, there is an option to pass keyword argument `debuginfo=:source`.  
```julia
julia> @code_lowered debuginfo=:source nextfib(3)
CodeInfo(
    @ none:2 within `nextfib'
1 ─ %1 = Main.one(n)
│   %2 = Main.one(n)
│        a = %1
└──      b = %2
    @ none:3 within `nextfib'
2 ┄ %5 = b < n
└──      goto #4 if not %5
    @ none:4 within `nextfib'
3 ─ %7 = b
│   %8 = a + b
│        a = %7
│        b = %8
└──      goto #2
    @ none:6 within `nextfib'
4 ─      return b
)
```

#### Code Typing
**Code typing** is the process in which the compiler attaches types to variables and tries to infer types of objects returned from called functions. If the compiler fails to infer the returned type, it will give the variable type `Any`, in which case a dynamic dispatch will be used in subsequent operations with the variable. Inspecting typed code is therefore important for detecting type instabilities (the process can be difficult and error prone, fortunately, new tools like `Jet.jl` may simplify this task). The output of typing can be inspected using `@code_typed` macro. If you know the types of function arguments, aka function signature, you can call directly function `InteractiveUtils.code_typed(nextfib, (typeof(3),))`.
```julia
julia> @code_typed nextfib(3)
CodeInfo(
1 ─      nothing::Nothing
2 ┄ %2 = φ (#1 => 1, #3 => %6)::Int64
│   %3 = φ (#1 => 1, #3 => %2)::Int64
│   %4 = Base.slt_int(%2, n)::Bool
└──      goto #4 if not %4
3 ─ %6 = Base.add_int(%3, %2)::Int64
└──      goto #2
4 ─      return %2
) => Int64
```

We can see that 
- some calls have been inlined, e.g. `one(n)` was replaced by `1` and the type was inferred as `Int`. 
-  The expression `b < n` has been replaced with its implementation in terms of the `slt_int` intrinsic ("signed integer less than") and the result of this has been annotated with return type `Bool`. 
- The expression `a + b` has been also replaced with its implementation in terms of the `add_int` intrinsic and its result type annotated as Int64. 
- And the return type of the entire function body has been annotated as `Int64`.
- The phi-instruction `%2 = φ (#1 => 1, #3 => %6)` is a **selector function**, which returns the value depending on from which branch do you come from. In this case, variable `%2` will have value 1, if the control was transfered from block `#1` and it will have value copied from variable `%6` if the control was transferreed from block `3` [see also](https://llvm.org/docs/LangRef.html#phi-instruction). The `φ` stands from *phony* variable.

When we have called `@code_lower`, the role of types of arguments was in selecting - via multiple dispatch - the appropriate function body among different methods. Contrary in `@code_typed`, the types of parameters determine the choice of inner methods that need to be called (again with multiple dispatch). This process can trigger other optimization, such as inlining, as seen in the case of `one(n)` being replaced with `1` directly, though here this replacement is hidden in the `φ` function. 

Note that the same view of the code is offered by the `@code_warntype` macro, which we have seen in the previous [lecture](@ref perf_lecture). The main difference from `@code_typed` is that it highlights type instabilities with red color and shows only unoptimized view of the code. You can view the unoptimized code with a keyword argument `optimize=false`:
```julia
julia> @code_typed optimize=false nextfib(3)
CodeInfo(
1 ─ %1 = Main.one(n)::Core.Const(1)
│   %2 = Main.one(n)::Core.Const(1)
│        (a = %1)::Core.Const(1)
└──      (b = %2)::Core.Const(1)
2 ┄ %5 = (b < n)::Bool
└──      goto #4 if not %5
3 ─ %7 = b::Int64
│   %8 = (a + b)::Int64
│        (a = %7)::Int64
│        (b = %8)::Int64
└──      goto #2
4 ─      return b
) => Int64
```

#### Lowering to LLVM IR
Julia uses the LLVM compiler framework to generate machine code. LLVM stands for low-level virtual machine and it is basis of many modern compilers (see [wiki](https://en.wikipedia.org/wiki/LLVM)).
We can see the textual form of code lowered to LLVM IR by invoking 
```julia
julia> @code_llvm debuginfo=:source nextfib(3)
;  @ REPL[10]:1 within `nextfib'
define i64 @julia_nextfib_890(i64 signext %0) {
top:
  br label %L2

L2:                                               ; preds = %L2, %top
  %value_phi  = phi i64 [ 1, %top ], [ %1, %L2 ]
  %value_phi1 = phi i64 [ 1, %top ], [ %value_phi, %L2 ]
;  @ REPL[10]:3 within `nextfib'
; ┌ @ int.jl:83 within `<'
   %.not = icmp slt i64 %value_phi, %0
; └
;  @ REPL[10]:4 within `nextfib'
; ┌ @ int.jl:87 within `+'
   %1 = add i64 %value_phi1, %value_phi
; └
;  @ REPL[10]:3 within `nextfib'
  br i1 %.not, label %L2, label %L8

L8:                                               ; preds = %L2
;  @ REPL[10]:6 within `nextfib'
  ret i64 %value_phi
}
```
LLVM code can be tricky to understand first, but one gets used to it. Notice references to the source code, which help with orientation. From the code above, we may infer
- code starts by jumping to label L2, from where it reads values of two variables to two "registers" `value_phi` and `value_phi1` (variables in LLVM starts with `%`). 
- Both registers are treated as `int64` and initialized by `1`. 
- `[ 1, %top ], [ %value_phi, %L2 ]` means that values are initialized as `1` if you come from the label `top` and as value `value_phi` if you come from `%2`. This is the LLVM's selector (phony `φ`).
- `icmp slt i64 %value_phi, %0` compares the variable `%value_phi` to the content of variable `%0`. Notice the anotation that we are comparing `Int64`.
- `%1 = add i64 %value_phi1, %value_phi` adds two variables `%value_phi1` and `%value_phi`. Note again than we are using `Int64` addition. 
- `br i1 %.not, label %L2, label %L8` implements a conditional jump depending on the content of `%.not` variable. 
- `ret i64 %value_phi` returns the value indicating it to be an `Int64`.

It is not expected you will be directly operating on the LLVM code, though there are libraries which does that. For example `Enzyme.jl` performs automatic differentiation of LLVM code, which has the benefit of being able to take a gradeint through `setdiff`.

#### Producing the native vode
**Native code** The last stage is generation of the native code, which Julia executes. The native code depends on the target architecture (e.g. x86, ARM). As in previous cases there is a macro for viewing the compiled code `@code_native`
```julia
julia> @code_native debuginfo=:source nextfib(3)
	.section	__TEXT,__text,regular,pure_instructions
; ┌ @ REPL[10]:1 within `nextfib'
	movl	$1, %ecx
	movl	$1, %eax
	nopw	(%rax,%rax)
L16:
	movq	%rax, %rdx
	movq	%rcx, %rax
; │ @ REPL[10]:4 within `nextfib'
; │┌ @ int.jl:87 within `+'
	addq	%rcx, %rdx
	movq	%rdx, %rcx
; │└
; │ @ REPL[10]:3 within `nextfib'
; │┌ @ int.jl:83 within `<'
	cmpq	%rdi, %rax
; │└
	jl	L16
; │ @ REPL[10]:6 within `nextfib'
	retq
	nopw	%cs:(%rax,%rax)
; └
```
and the output is used mainly for debugging / inspection. 

## Looking around the language
Language introspection is very convenient for investigating, how things are implemented and how they are optimized / compiled to the native code.

!!! note "Reminder `@which`"
	Though we have already used it quite a few times, recall the very useful macro `@which`, which identifies the concrete function called in a function call. For example `@which mapreduce(sin, +, [1,2,3,4])`. Note again that the macro here is a convenience macro to obtain types of arguments from the expression. Under the hood, it calls `InteractiveUtils.which(function_name, (Base.typesof)(args...))`. Funnily enough, you can call `@which InteractiveUtils.which(+, (Base.typesof)(1,1))` to inspect, where `which` is defined.

### Broadcasting
Broadcasting is not a unique concept in programming languages (Python/Numpy, MATLAB), however its implementation in Julia allows to easily fuse operations. For example 
```julia
x = randn(100)
sin.(x) .+ 2 .* cos.(x) .+ x
```
is all computed in a single loop. We can inspect, how this is achieved in the lowered code:
```julia
julia> Meta.@lower sin.(x) .+ 2 .* cos.(x) .+ x
:($(Expr(:thunk, CodeInfo(
    @ none within `top-level scope'
1 ─ %1 = Base.broadcasted(sin, x)
│   %2 = Base.broadcasted(cos, x)
│   %3 = Base.broadcasted(*, 2, %2)
│   %4 = Base.broadcasted(+, %1, %3)
│   %5 = Base.broadcasted(+, %4, x)
│   %6 = Base.materialize(%5)
└──      return %6
))))
```
Notice that we have not used the usual `@code_lowered` macro, because the statement to be lowered is not a function call. In these cases, we have to use `@code_lowered`, which can handle more general program statements. On these cases, we cannot use `@which` either, as that applies to function calls only as well.

### Generators
```julia
Meta.@lower [x for x in 1:4]
:($(Expr(:thunk, CodeInfo(
    @ none within `top-level scope'
1 ─ %1 = 1:4
│   %2 = Base.Generator(Base.identity, %1)
│   %3 = Base.collect(%2)
└──      return %3
))))
```
from which we see that the `Generator` is implemented using the combination of a `Base.collect`, which is a function collecting items of a sequence and `Base.Generator(f,x)`, which implements an iterator, which applies function `f` on elements of `x` over which is being iterated. So an almost magical generators have instantly lost their magic.

### Closures
```julia
adder(x) = y -> y + x

julia> @code_lowered adder(5)
CodeInfo(
1 ─ %1 = Main.:(var"#8#9")
│   %2 = Core.typeof(x)
│   %3 = Core.apply_type(%1, %2)
│        #8 = %new(%3, x)
└──      return #8
)
```

### The effect of type-instability
```julia
struct Wolf
	name::String
	energy::Int
end

struct Sheep
	name::String
	energy::Int
end

sound(wolf::Wolf) = println(wolf.name, " has howled.")
sound(sheep::Sheep) = println(sheep.name, " has baaed.")
stable_pack   = (Wolf("1", 1), Wolf("2", 2), Sheep("3", 3))
unstable_pack = [Wolf("1", 1), Wolf("2", 2), Sheep("3", 3)]
@code_typed map(sound, stable_pack)
@code_typed map(sound, unstable_pack)
```
!!! info 
	## Cthulhu.jl
	`Cthulhu.jl` is a library (tool) which simplifies the above, where we want to iteratively dive into functions called in some piece of code (typically some function). `Cthulhu` is different from te normal debugger, since the debugger is executing the code, while `Cthulhu` is just lower_typing the code and presenting functions (with type of arguments inferred).

```julia
using Cthulhu
@descend map(sound, unstable_pack)
```

## General notes on metaprogramming
According to an excellent [talk](https://www.youtube.com/watch?v=mSgXWpvQEHE) by Steven Johnson, you should use metaprogramming sparingly, because on one hand it's very powerful, but  on the other it is generally difficult to read and it can lead to unexpected errors. Julia allows you to interact with the compiler at two different levels.
1. After the code is parsed to AST, you can modify it directly or through **macros**.
2. When SSA form is being typed, you can create custom functions using the concept of **generated functions**.

More functionalities are coming through the [JuliaCompilerPlugins](https://github.com/JuliaCompilerPlugins) project, but we will not talk about them (yet). 

## What is Quotation?
When we are doing metaprogramming, we need to somehow tell the compiler that the next block of code is not a normal block of code to be executed, but that it should be interpreted as data and in any sense it should not be evaluated. **Quotation** refers to exactly this syntactic sugar. In Julia, quotation is achieved either through `:(...)` or `quote ... end`.

Notice the difference between
```julia
1 + 1 
```
and 
```julia
:(1 + 1)
```

The type returned by the quotation depends on what is quoted. Observe the returned type of the following quoted code
```julia
:(1)			|> typeof
:(:x)		 	|> typeof
:(1 + x) 		|> typeof
quote
    1 + x
    x + 1
end 			|> typeof
```
All of these snippets are examples of the quoted code, but only `:(1 + x)` and the quote block produce objects of type `Expr`. An interesting return type is the `QuoteNode`, which allows to insert piece of code which should contain elements that should not be interpolated. Most of the time, quoting returns `Expr`essions.

## Expressions
Abstract Syntax Tree, the output of Julia's parser, is expressed using Julia's own datastructures, which means that you can freely manipulate it (and constructed) from the language itself. This property is called **homoiconicity**. Julia's compiler allows you to intercept compilation just after it has parsed the source code. Before we will take advantage of this power, we should get familiar with the strucute of the AST.

The best way to inspect the AST is through the combination 
- `Meta.parse,`  which parses the source code to AST, 
- `dump` which print AST to terminal, 
- `eval` which evaluates the AST within the current module.

Let's start by investigating a very simple statement `1 + 1`, whose AST can be constructed either by `Meta.parse("1 + 1")` or `:(1 + 1)` or `quote 1+1 end` (the last one includes also the line information metadata).
```julia
julia> p = :(1+1)
:(1 + 1)

julia> typeof(p)
Expr

julia> dump(p)
Expr
  head: Symbol call
  args: Array{Any}((3,))
    1: Symbol +
    2: Int64 1
    3: Int64 1
```
The parsed code `p` is of type `Expr`, which according to Julia's help[^2] is *a type representing compound expressions in parsed julia code (ASTs). Each expression consists: of a head Symbol identifying which kind of expression it is (e.g. a call, for loop, conditional statement, etc.), and subexpressions (e.g. the arguments of a call). The subexpressions are stored in a Vector{Any} field called args.* If you recall the figure above, where AST was represented as a tree, `head` gives each node the name name `args` are either some parameters of the node, or they point to childs of that node. The interpretation of the node depends on the its type stored in head (note that the word type used here is not in the Julia sense).

[^2]: [https://docs.julialang.org/en/v1/base/base/#Core.Expr](https://docs.julialang.org/en/v1/base/base/#Core.Expr)

!!! info "`Symbol` type"
	When manipulations of expressions, we encounter the term `Symbol`. `Symbol` is the smallest atom from which the program (in AST representation) is built. It is used to identify an element in the language, for example variable, keyword or function name. Symbol is not a string, since string represents itself, whereas `Symbol` can represent something else (a variable). An illustrative example[^3] goes as follows.
	```julia
	julia> eval(:foo)
	ERROR: foo not defined

	julia> foo = "hello"
	"hello"

	julia> eval(:foo)
	"hello"

	julia> eval("foo")
	"foo"
	```
	which shows that what the symbol `:foo` evaluates to depends on what – if anything – the variable `foo` is bound to, whereas "foo" always just evaluates to "foo".

	Symbols can be constructed either by prepending any string with `:` or by calling `Symbol(...)`, which concatenates the arguments and create the symbol out of it. All of the following are symbols
	```julia
	julia> :+
	:+

	julia> :function
	:function

	julia> :call
	:call

	julia> :x
	:x

	julia> Symbol(:Very,"_twisted_",:symbol,"_definition")
	:Very_twisted_symbol_definition

	julia> Symbol("Symbol with blanks")
	Symbol("Symbol with blanks")
	```
	Symbols therefore allows us to operate with a piece of code without evaluating it.

	In Julia, symbols are "interned strings", which means that compiler attaches each string a unique identifier (integer), such that it can quickly compare them. Compiler uses Symbols exclusively and the important feature is that they can be quickly compared. This is why people like to use them as keys in `Dict`.

[^3]: An example provided by Stefan Karpinski [https://stackoverflow.com/questions/23480722/what-is-a-symbol-in-julia](https://stackoverflow.com/questions/23480722/what-is-a-symbol-in-julia)

!!! info "`Expr`essions"
	From Julia's help[^2]:

	`Expr(head::Symbol, args...)`

	A type representing compound expressions in parsed julia code (ASTs). Each expression consists of a head `Symbol` identifying which kind of expression it is (e.g. a call, for loop, conditional statement, etc.), and subexpressions (e.g. the arguments of a call).
	The subexpressions are stored in a `Vector{Any}` field called args. 

	The expression is simple yet very flexible. The head `Symbol` tells how the expression should be treated and arguments provide all needed parameters. Notice that the structure is also type-unstable. This is not a big deal, since the expression is used to generate code, hence it is not executed repeatedly.

## Construct code from scratch
Since `Expr` is a Julia structure, we can construct it manually as we can construct any other structure
```julia
julia> Expr(:call, :+, 1 , 1) |> dump
Expr
  head: Symbol call
  args: Array{Any}((3,))
    1: Symbol +
    2: Int64 1
    3: Int64 1
```
yielding to the same structure as we have created above. 
Expressions can be evaluated using `eval`, as has been said. to programmatically evaluate our expression, let's do 
```julia
e = Expr(:call, :+, 1, 1)
eval(e)
```
We are free to use variables (identified by symbols) inside the expression 
```julia
e = Expr(:call, :+, :x, 5)
eval(e)
```
but unless they are not defined within the scope, the expression cannot produce a meaningful result
```julia
x = 3
eval(e)
```

```julia
:(1 + sin(x)) == Expr(:call, :+, 1, Expr(:call, :sin, :x))
```

Since the expression is a Julia structure, we are free to manipulate it. Let's for example substitutue `x` in  `e = :(x + 5)` with `2x`.
```julia
e = :(x + 5)
e.args = map(e.args) do a 
	a == :x ?  Expr(:call, :*, 2, :x) : a 
end
```

or 
```julia
e = :(x + 5)
e.args = map(e.args) do a 
	a == :x ? :(2*x) : a 
end
```
and verify that the results are correct.
```julia
julia> dump(e)
Expr
  head: Symbol call
  args: Array{Any}((3,))
    1: Symbol +
    2: Expr
      head: Symbol call
      args: Array{Any}((3,))
        1: Symbol *
        2: Int64 2
        3: Symbol x
    3: Int64 5

julia> eval(e)
11
```
As already mentioned, the manipulation of Expression can be arbitrary. In the above example, we have been operating directly on the arguments. But what if `x` would be deeper in the expression, as for example in `2(3 + x) + 2(2 - x) `? We can implement the substitution using multiple dispatch as we would do when implementing any other function in Julia.
```julia
replace_x(x::Symbol) = x == :x ? :(2*x) : x
replace_x(e::Expr) = Expr(e.head, map(replace_x, e.args)...)
replace_x(u) = u
```
which works as promised.
```julia
julia> e = :(2(3 + 2x) + 2(2 - x))
:(2 * (3 + x) + 2 * (2 - x))
julia> f = replace_x(e)
:(2 * (3 + 2x) + 2 * (2 - 2x))
```

or we can replace the `sin` function
```julia
replace_sin(x::Symbol) = x == :sin ? :cos : x
replace_sin(e::Expr) = Expr(e.head, map(replace_sin, e.args)...)
replace_sin(u) = u
```
```julia
replace_sin(:(1 + sin(x)))
```
Sometimes, we want to operate on a block of code as opposed to single line expressions. Recall that a block of code is defined-quoted with `quote ... end`. Let us see how `replace_x` can handle the following example:
```julia
e = quote 
	a = x + 3
	b = 2 - x
	2a + 2b
end
```

```julia
julia> replace_x(e) |> Base.remove_linenums!
quote
    a = 2x + 3
    b = 2 - 2x
    2a + 2b
end

julia> replace_x(e) |> eval
10
```

### Brittleness of code manipulation
When we are manipulating the AST or creating new expressions from scratch, there is no **syntactic** validation performed by the parser. It is therefore very easy to create AST which does not make any sense and cannot be compiled. We have already seen that we can refer to variables that were not defined yet (this makes perfect sense). The same goes with functions (which also makes a lot of sense).
```julia
e = :(g() + 5)
eval(e)
g() = 5
eval(e)
```
But we can also introduce keywords which the language does not know. For example 
```julia
e = Expr(:my_keyword, 1, 2, 3)
:($(Expr(:my_keyword, 1, 2, 3)))

julia> e.head
:my_keyword

julia> e.args
3-element Vector{Any}:
 1
 2
 3

julia> eval(e)
ERROR: syntax: invalid syntax (my_keyword 1 2 3)
Stacktrace:
 [1] top-level scope
   @ none:1
 [2] eval
   @ ./boot.jl:360 [inlined]
 [3] eval(x::Expr)
   @ Base.MainInclude ./client.jl:446
 [4] top-level scope
   @ REPL[8]:1
```
notice that error is not related to undefined variable / function, but the invalid syntax. This also demonstrates the role of `head` in `Expr`. More on Julia AST can be found in the developer [documentation](https://docs.julialang.org/en/v1/devdocs/ast/#Julia-ASTs).

### Alternative way to look at code
```julia
Meta.parse("x[3]") |> dump
```
We can see a new Symbol `ref` as a head and the position `3` of variable `x`.

```julia
Meta.parse("(1,2,3)") |> dump
```

```julia
Meta.parse("1/2/3") |> dump
```



<!-- ### Algebraic expansion with macro
```julia
function match_sin_xy(ex::Expr) 
	ex.head != :call && return(false)
	length(ex.args) != 2 && return(false)
	ex.args[1] != :sin && return(false)
	!(ex.args[2] isa Expr) && return(false)
	ix = ex[2]
	ix.head != :call && return(false)
	length(ix.args) != 3 && return(false)
	ix.args[1] != :+ && return(false)
end

function rewrite_sin_xy(ex::Expr) 
	ex.head != :call && return(false)
	length(ex.args) != 2 && return(false)
	ex.args[1] != :sin && return(false)
	!(ex.args[2] isa Expr) && return(false)
	ix = ex[2]
	ix.head != :call && return(false)
	length(ix.args) != 3 && return(false)
	ix.args[1] != :+ && return(false)
end

expand_sin_xy(:(sin(x + 2*(z + y))))
```
-->
## Code generation

### Using metaprogramming in inheritance by encapsulation
Recall that Julia (at the moment) does not support inheritance, therefore the only way to adopt functionality of some object and extend it is through *encapsulation*. Assuming we have some object `T`, we wrap that object into a new structure.
Let's work out a concrete example, where we define the our own matrix. 
```julia
struct MyMatrix{T} <: AbstractMatrix{T}
	x::Matrix{T}
end
```
Now, to make it useful, we should define all the usual methods, like `size`, `length`, `getindex`, `setindex!`, etc. We can list methods defined with `Matrix` as an argument `methodswith(Matrix)` (recall this will load methods that are defined with currently loaded libraries). Now, we would like to overload them. To minimize the written code, we can write
```julia
import Base: setindex!, getindex, size, length
for f in [:setindex!, :getindex, :size, :length]
	eval(:($(f)(A::MyMatrix, args...) = $(f)(A.x, args...)))
end
```
which we can verify now that it works as expected 
```julia
julia> a = MyMatrix([1 2 ; 3 4])
2×2 MyMatrix{Int64}:
 1  2
 3  4

julia> a[4]
4

julia> a[3] = 0
0

julia> a
2×2 MyMatrix{Int64}:
 1  0
 3  4
```
In this way, Julia acts as its own preprocessor.
The above look can be equally written as 
```julia
for f in [:setindex!, :getindex, :size, :length]
	s = "Base.$(f)(A::MyMatrix, args...) = $(f)(A.x, args...)"
	println(s)
	eval(Meta.parse(s))
end
```
for f in [:setindex!, :getindex, :size, :length]
	@eval $(f)(A::MyMatrix, args...) = $(f)(A.x, args...)
end

Notice that we have just hand-implemented parts of `@forward` macro from [MacroTools](https://github.com/FluxML/MacroTools.jl/blob/master/src/examples/forward.jl), which does exactly this.

<!-- Should I mention the world clock age and the effect of eval in global scope -->

---

# Resources
- Julia's manual on [metaprogramming](https://docs.julialang.org/en/v1/manual/metaprogramming/)
- David P. Sanders' [workshop @ JuliaCon 2021](https://www.youtube.com/watch?v=2QLhw6LVaq0) 
- Steven Johnson's [keynote talk @ JuliaCon 2019](https://www.youtube.com/watch?v=mSgXWpvQEHE)
- Andy Ferris's [workshop @ JuliaCon 2018](https://www.youtube.com/watch?v=SeqAQHKLNj4)
- [From Macros to DSL](https://github.com/johnmyleswhite/julia_tutorials) by John Myles White 
- Notes on [JuliaCompilerPlugin](https://hackmd.io/bVhb97Q4QTWeBQw8Rq4IFw?both#Julia-Compiler-Plugin-Project)