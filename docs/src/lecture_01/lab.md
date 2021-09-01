# Lab 01: Introduction to Julia
This lab should get everyone up to speed in the basics of Julia's installation, syntax and basic coding. For more detailed introduction you can check out Lectures 1-3 of the bachelor [course](https://juliateachingctu.github.io/Julia-for-Optimization-and-Learning/stable/).

## Testing Julia installation
In order to proceed further let's run a simple script to see, that the setup described in chapter [Installation](@ref install) is working properly.
After spawning a terminal run this command:
```bash
julia ./test_setup.jl
```
The script does the following 
- "Tests" if Julia is added to path and can be run with `julia` command from anywhere
- Prints Hello World and Julia version info
- Creates an environment configuration files
- Installs a basic pkg called BenchmarkTools, which we will use for benchmarking a simple function later in the labs.

There are some quality of life improvements over long term support versions of Julia and thus for the course of these lectures we will use the latest stable release of Julia 1.6.x.


## Polynomial evaluation example
Let's consider a common mathematical example for evaluation of nth-degree polynomial
```math
f(x) = a_{n}x^{n} + a_{n-1}x^{n-1} + \dots + a_{0}x^{0},
```
where $x \in \mathbb{R}$ and $\vec{a} \in \mathbb{R}^{n+1}$.

The simplest way of writing this is just realizing that essentially the function $f$ is really implicitly containing argument $\vec{a}$, i.e. $f \equiv f(\vec{a}, x)$, yielding the following function

```@example 1
function polynomial(a, x)
    accumulator = 0
    for i in length(a):-1:1
        accumulator += x^(i-1) * a[i] # ! 1-based indexing for arrays
    end
    return accumulator
end
nothing #hide
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Evaluate the code of the function called `polynomial` in Julia REPL and evaluate the function itself with the following arguments.
```@example 1
a = [-19, 7, -4, 6] # list coefficients a from a^0 to a^n
x = 3               # point of evaluation
nothing #hide
```

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

The simplest way is to just copy&paste into an already running terminal manually. As opposed to the default Python REPL, Julia can deal with the blocks of code and different indentation much better without installation of an `ipython`-like REPL. There are ways to make this much easier in different text editors/IDEs:
- `VSCode` - when using Julia extension, by default `Ctrl+Enter` will spawn Julia REPL, when a `.jl` file is opened
- `Sublime Text` - `Send Code` pkg (works well with Linux terminal or tmux, support for Windows is poor)
- `Vim` - there is a Julia language [plugin](https://github.com/JuliaEditorSupport/julia-vim), which can be combine with [vimcmdline](https://github.com/jalvesaq/vimcmdline)

Either way, you should see the following:
```@repl 1
function polynomial(a, x)
    accumulator = 0
    for i in length(a):-1:1
        accumulator += x^(i-1) * a[i] # ! 1-based indexing for arrays
    end
    return accumulator
end
```

Similarly we enter the arguments of the function `a` and `x`:
```@repl 1
a = [-19, 7, -4, 6]
x = 3
```


Function call intuitively takes the name of the function with round brackets as arguments, i.e. works in the same way as majority of programming languages. The result is printed unless a `;` is added at the end of the statement.
```@repl 1
polynomial(a, x)    # function call
```

```@raw html
</p></details>
```

Thanks to the high level nature of Julia language it is often the case that examples written in pseudocode are almost directly rewritable into the language itself without major changes and the code can be thus interpreted easily.

![polynomial_explained](./polynomial.svg)

The indentation is not necessary as opposed to other languages such as Python, due to the existence of the `end` keyword, however it is strongly recommended to use it, see [style guide](https://docs.julialang.org/en/v1/manual/style-guide/#Style-Guide). Furthermore the return keyword can be omitted if the last line being evaluated contains the result, unless the line ends with `;`.


Though there are libraries/IDEs that allow us to step through Julia code (`Rebugger.jl` [link](https://github.com/timholy/Rebugger.jl) and `VSCode` [link](https://www.julia-vscode.org/docs/stable/userguide/debugging/)), we can (having defined the arguments with the same name as inside the actual function) evaluate pieces of code separately. 

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```

Evaluate the following pieces of code and check their type with `typeof` function, e.g. `typeof(a)` or `typeof([-19, 7, -4, 6])`

**BONUS**: Try to "call for help" by accessing the build in help terminal by typing `?` followed by a keyword to explain. Use this for basic functions such as `length`, `typeof`, `^`.
```julia
a = [-19, 7, -4, 6]
x = 3
accumulator = 0
length(a):-1:1

i = length(a)
accumulator += x^(i-1) * a[i]
accumulator

polynomial
^
*
```

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
When defining a variable through an assignment we get the representation of the right side, again this is different from the default 
behavior in Python, where the output of `a = [-19, 7, -4, 6]`, prints nothing. In Julia REPL we get the result of the `display` function.

```@repl 1
julia> a = [-19, 7, -4, 6]
julia> display(a) # should return the same thing as the line above
```
As you can see, the string that is being displayed contains information about the contents of a variable along with it's type in this case this is a `Vector` of `Int` types. Which can be checked further with the `typeof` function:
```@repl 1
typeof(a)
```
In most cases variables store just a reference to a place in memory either stack/heap (exceptions are primitive types such as `Int`, `Float`) and therefore creating an array `a`, storing the reference in `b` with `b = a` and changing elements of `b`, e.g. `b[1] = 2`, changes also the values in `a`.

The other two assignments are exactly the same as they both generate an instance of `Int` type with different values. Though now one has to call for hell the `typeof` function, because by default this information is omitted in the display of simple types.
```@repl 1
x = 3
accumulator = 0
typeof(x), typeof(accumulator)
```

The next expression creates an instance of a range, which are *inclusive* in Julia, i.e. containing number from start to end - in this case running from `4` to `1` with negative step `-1`, thus counting down.
```@repl 1
length(a):-1:1
typeof(length(a):-1:1)
```

Let's confirm the fact that the update operator `+=` really does update the variable `accumulator` by running the following
```@repl 1
i = length(a) # 
accumulator += x^(i-1) * a[i]
accumulator
```
Notice that evaluating a variable, which can be used instead of the return keyword at the end of a function.


We have already seen the output of evaluating `polynomial` function name in the REPL
By creating the function `polynomial` we have defined a variable `polynomial`, that from now on always refers to a function and cannot be redefined with a different type.
```@repl 1
polynomial
```
This is cause by the fact that each function defines essentially a new type, the same like `Int ~ Int64` or `Vector{Int}`.
```@repl 1
typeof(polynomial)
```
You can check that it is a subtype of the `Function` abstract type, with the subtyping operator `<:`
```@repl 1
typeof(polynomial) <: Function
```
These concepts will be expanded further in the type [lecture](@ref type_lecture), however for now note that this construction is quite useful for example if we wanted to create derivative rules for our function `derivativeof(::typeof(polynomial), ...)`.

Looking at the last two expersions `+`, `*`, we can see that in Julia, operators are also functions. 
```@repl 1
+
*
```
The main difference from our `polynomial` function is that there are multiple methods, for each of these functions. Each one of the methods coresponds to a specific combination of arguments, for which the function can be specialized to. You can see the list by calling a `methods` function:
```@repl 1
methods(+)
```
One other notable difference is that these functions allow using both infix and postfix notation `a + b` and `+(a,b)`, which is a speciality of elementary functions such as arithmetic operators or set operation such as `∩, ∪, ∈`.


**BONUS**: Accessing help terminal `?` and looking up a keyword, searches for documentation of individual methods/functions in the source code. When creating a pkg, it is desirable to create so called `docstrings` for each method that is going to be exported. `docstrings` are multiline strings written above a function. More on this in [lecture](@ref pkg_lecture) on pkg development.

```julia
"""
    polynomial(a, x)

Returns value of a polynomial with coefficients `a` at point `x`.
"""
function polynomial(a, x)
    # function body
end
```

```@raw html
</p></details>
```

As the arguments of the `polynomial` functions are untyped, i.e. they do not specify the allowed types like for example `polynomial(a, x::Number)` does, the following exercise explores how wide range of arguments does the

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
What happens if you call our polynomial function with with the following coefficients?

```@example 1
af = [-19.0, 7.0, -4.0, 6.0]
at = (-19, 7, -4, 6)
ant = (a₀ = -19, a₁ = 7, a₂ = -4, a₃ = 6)
a2d = [-19 -4; 7 6]
ach = ['1', '2', '3', '4']
ac = [2i^2 + 1 for i in -2:1]
ag = (2i^2 + 1 for i in -2:1)
nothing #hide
```

Check first the types of each of these coefficients by calling `typeof` and `eltype`.

**BONUS**: In the case of `ag`, use the `collect` function to get the desirable result. What does it do? Check again the type of the result.
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@repl 1
typeof(af), eltype(af)
polynomial(af, x)
```
As opposed to the basic definition of `a` type the array is filled with `Float64` types and the resulting value gets promoted as well to the `Float64`.


```@repl 1
typeof(at), eltype(at)
polynomial(at, x)
```
With round brackets over a fixed length vector we get the `Tuple` type, which is a fixed size, so called immutable "array" of a fixed size (its elements cannot be changed, unless initialized from scratch). Each element can be of a different type, but here we have only one and thus the `Tuple` is aliased into `NTuple`. There are some performance benefits for using immutable structure, which will be discussed [later](@ref type_lecture) or [even later](@ref perf_lecture).


Defining `key=value` pairs inside round brackets creates a structure called `NamedTuple`, which has the same properties as `Tuple` and furthermore it's elements can be conveniently accessed by dot syntax, e.g. `ant.a₀`.
```@repl 1
typeof(ant), eltype(ant)
polynomial(ant, x)
```

Defining a 2D array is a simple change of syntax, which initialized a matrix row by row separated by `;` with spaces between individual elements. The function works in the same way because linear indexing works in 2d arrays in the column major order.
```@repl 1
typeof(a2d), eltype(a2d)
polynomial(a2d, x)
```

Consider the vector/array of characters, which themselves have numeric values (you can check by converting them to Int `Int('1')` or `convert(Int, 'l')`). In spite of that, our untyped function cannot process such input, as there isn't an operation/method that would allow  multiplication of `Char` and `Int` type. Julia tries to promote the argument types to some common type, however checking the `promote_type(Int, Char)` returns `Any` (union of all types), which tells us that the conversion is not possible automatically.
```@repl 1
typeof(ach), eltype(ach)
polynomial(ach, x)
```
In the stacktrace we can see the location of each function call. If we include the function `polynomial` from some file `poly.jl` using `include("poly.jl")`, we will see that the location changes from `REPL[X]:10` to the the actual file name.

The next example shows so called array comprehension syntax, where we define and array of known length using and for loop iteration. Resulting array/vector has integer elements, however even mixed type is possible yielding `Any`, if there isn't any other common supertype to `promote` every entry into. (Use `?` to look what `promote` and `promote_type` does.)
```@repl 1
typeof(ac), eltype(ac)
polynomial(ac, x)
```

By swapping square brackets for round we have defined so called generator/iterator, which as opposed to the previous example does not allocate an array, only the structure that produces it. You may notice that the element type in this case is `Any`, which means that a function using such generator as an argument cannot specialize based on the type and has to infer it every time an element is generated/returned. We will touch on how this affects performance in one of the later [lectures](@ref perf_lecture).
```@repl 1
typeof(ag), eltype(ag)
polynomial(ag, x)
```

**BONUS**: In general generators may have unknown length, this can be useful for example in batch processing of files, where we do not know beforehand how many files are in a given folder. However the problem here originated from a missing indexing operation `getindex`, which can be easily solved by collecting the generator with `collect` and thus transforming it into and array.
```@repl 1
agc = ag |> collect # pipe syntax, equivalent to collect(ag)
typeof(agc), eltype(agc)
```

You can see now that `eltype` is no longer `Any`, as a proper type for the whole container has been found in the `collect` function, however we have lost the advantage of not allocating an array.

```@raw html
</p></details>
```

## Extending/limiting the polynomial example
We have seen the 


```
```


## How to use code from other people
The script that we have run at the beginning of this lab has created a folder `test` with the following files.
```
./test/
    ├── Manifest.toml
    ├── Project.toml
    └── src
        └── test.jl
```
Every folder with a toml file called `Project.toml`, can be used by Julia's pkg manager into setting so called environment. Each of these environments has a specific name, unique identifier and most importantly a list of pkgs to be installed. Setting up or more often called activating an environment can be done either before starting Julia itself by running julia with the `--project XXX` flag or from withing the Julia REPL, by switching to Pkg mode with `]` key (similar to the help mode activated by pressing `?`) and running command `activate`.

So far we have used the general environment, which by default does not come with any 3rd party packages and includes only the base and standard libraries - [already](https://docs.julialang.org/en/v1/base/arrays/) [quite](https://docs.julialang.org/en/v1/base/multi-threading/) [powerful](https://docs.julialang.org/en/v1/stdlib/Distributed/) [on its own](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/). 

In order to find which environment is currently active, run the following:
```julia
] status
```
The output of such command usually indicates the general environment located at `.julia/` folder (`${HOME}/.julia/` or `${APPDATA}/.julia/` in case of Unix/Windows based systems respectively)
```julia
(@v1.6) pkg> status
Status `~/.julia/environments/v1.6/Project.toml` (empty project)
```
Generally one should avoid working in the general environment, with the exception using some generic pkgs, such as `PkgTemplates.jl`, which is used for generating pkg templates/folder structure like the one above ([link](https://github.com/invenia/PkgTemplates.jl)), more on this in the [lecture](@ref pkg_lecture) on pkg development. 


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Activate the test environment inside `./test` and check that the `BenchmarkTools` package has been installed. Use `BenchmarkTools` pkg's `@btime` to benchmark our `polynomial` function with the following arguments.
```@example 1
aexp = ones(10) ./ factorial.(0:9)
x = 1.1
nothing #hide
```

**HINTS:**
- In pkg mode use the command `activate` and `status` to check the presence. 
- In order to import the functionality from other package, lookup the keyword `using` in the repl help mode `?`. 
- The functionality that we want to use is the `@btime` macro (it acts almost like a function but with a different syntax `@macro arg1 arg2 arg3 ...`). More on macros in the corresponding [lecture](@ref macro_lecture).

**BONUS**: Compare the output of `polynomial(aexp, x)` with the value of `exp(x)`, which it approximates.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

There are other options to import a function/macro from a different package, however for now let's keep it simple with the `using Module` syntax, that brings to the REPL, all the variables/function/macros exported by the `BenchmarkTools` pkg. If `@btime` is exported, which it is, it can be accessed without specification i.e. just by calling `@btime` without the need for `BenchmarkTools.@btime`. More on the architecture of pkg/module loading in the package developement [lecture](@ref pkg_lecture).
```@repl 1
using BenchmarkTools
@btime polynomial(aexp, x)
```
The output gives us the time of execution averaged over multiple runs (the number of samples is defined automatically based on run time) as well as the number of allocations and the output of the function, that is being benchmarked.


**BONUS**: The difference between our approximation and the "actual" function value computed as a difference of the two. 
```@repl 1
polynomial(aexp, x) - exp(x)
```
The apostrophes in the previous sentece are on purpose, because implementation of `exp` also relies too on a finite sum, though much more sophisticated than the basic Taylor expansion.

```@raw html
</p></details>
```


## Useful resources
- Getting Started tutorial from JuliaLang documentation - [Docs](https://docs.julialang.org/en/v1/manual/getting-started/)
- Converting syntax between MATLAB ↔ Python ↔ Julia - [Cheatsheet](https://cheatsheets.quantecon.org/)
- Bachelor course for refreshing your knowledge - [Course](https://juliateachingctu.github.io/Julia-for-Optimization-and-Learning/stable/)
- Stylistic conventions - [Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/#Style-Guide)
- Reserved keywords - [List](https://docs.julialang.org/en/v1/base/base/#Keywords)


### Various errors and how to read them
This section summarizes most commonly encountered types of errors in Julia and shows how to read them. [Documentation](https://docs.julialang.org/en/v1/base/base/#Errors) contains the complete list and each individual error can be queried against the `?` mode of the REPL.

#### `MethodError`
This type of error is most commonly thrown by Julia's multiple dispatch system with a message like `no method matching X(args...)`, seen in two examples bellow.
```@repl 2
2 * 'a'                       # many candidates
getindex((i for i in 1:4), 3) # no candidates
```
Both of these examples have a short stacktrace, showing that the execution failed on the top most level in `REPL`, however if this code is a part of some function in a separate file, the stacktrace will reflect it. What this error tells us is that the dispatch system could not find a method for a given function, that would be suitable for the type of arguments, that it has been given. In the first case Julia offers also a list of candidate methods, that match at least some of the arguments

When dealing with basic Julia functions and types, this behavior can be treated as something given and though one could locally add a method for example for multiplication of `Char` and `Int`, there is usually a good reason why Julia does not support such functionality by default. On the other hand when dealing with user defined code, this error may suggest the developer, that either the functions are too strictly typed or that another method definition is needed in order to satisfy the desired functionality.

#### `InexactError`
This type of error is most commonly thrown by the type conversion system (centered around `convert` function), informing the user that it cannot exactly convert a value of some type to match arguments of a function being called.
```@repl 2
Int(1.2)                      # root cause
append!([1,2,3], 1.2)         # same as above but shows the root cause deeper in the stack trace
```
In this case the function being `Int` and the value a floating point. The second example shows `InexactError` may be caused deeper inside an inconspicuous function call, where we want to extend an array by another value, which is unfortunately incompatible.

#### `ArgumentError`
As opposed to the previous two errors, `ArgumentError` can contain user specified error message and thus can serve multiple purposes. It is however recommended to throw this type of error, when the parameters to a function call do not match a valid signature, e.g. when `factorial` were given negative or non-integer argument (note that this is being handled in Julia by multiple dispatch and specific `DomainError`).

This example shows a concatenation of two 2d arrays of incompatible sizes 3x3 and 2x2.
```@repl 2
hcat(ones(3,3), zeros(2,2))
```

#### `KeyError`
This error is specific to hash table based objects such as the `Dict` type and tells the user that and indexing operation into such structure tried to access or delete a non-existent element.
```@repl 2
d = Dict(:a => [1,2,3], :b => [1,23])
d[:c]
```

#### `TypeError`
Type assertion failure, or calling an intrinsic function (inside LLVM, where code is strictly typed) with incorrect argument type. In practice this error comes up most often when comparing value of a type against the `Bool` type as seen in the example bellow.
```@repl 2
if 1 end                # calls internally typeassert(1, Bool)
typeassert(1, Bool)
```
In order to compare inside conditional statements such as `if-elseif-else` or the ternary operator `x ? a : b` the condition has to be always of `Bool` type, thus the example above can be fixed by the comparison operator: `if 1 == 1 end` (in reality either the left or the right side of the expression contains an expression or a variable to compare against).

#### `UndefVarError`
While this error is quite self-explanatory, the exact causes are often quite puzzling for the user. The reason behind the confusion is to do with *code scoping*, which comes into play for example when trying to access a local variable from outside of a given function or just updating a global variable from within a simple loop. 

In the first example we show the former case, where variable is declared from within a function and accessed from outside afterwards.
```@repl 2
function plusone(x)
    uno = 1
    return x + uno
end
uno # defined only within plusone
```

Unless there is variable `I_am_not_defined` in the global scope, the following should throw an error.
```@repl 2
I_am_not_defined
```
Often these kind of errors arise as a result of bad code practices, such as long running sessions of Julia having long forgotten global variables, that do not exist upon new execution (this one in particular has been addressed by the authors of the reactive Julia notebooks [Pluto.jl](https://github.com/fonsp/Pluto.jl)).

For more details on code scoping we recommend particular places in the bachelor course lectures [here](https://juliateachingctu.github.io/Julia-for-Optimization-and-Learning/stable/lecture_02/scope/#Soft-local-scope) and [there](https://juliateachingctu.github.io/Julia-for-Optimization-and-Learning/stable/lecture_03/scope/#Scope-of-variables).