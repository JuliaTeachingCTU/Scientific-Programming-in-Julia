# [Lab 07: Macros](@id macro_lab)
A little reminder from the [lecture](@ref macro_lecture), a macro in its essence is a function, which 
1. takes as an input an expression (parsed input)
2. modifies the expressions in arguments
3. inserts the modified expression at the same place as the one that is parsed.

In this lab we are going to use what we have learned about manipulation of expressions and explore avenues of where macros can be useful
- convenience (`@repeat`, `@show`)
- performance critical code generation (`@poly`)
- alleviate tedious code generation (`@species`, `@eats`)
- just as a syntactic sugar (`@ecosystem`)

## Show macro
Let's start with dissecting "simple" `@show` macro, which allows us to demonstrate advanced concepts of macros and expression manipulation.
```@repl lab07_show
x = 1
@show x + 1
let y = x + 1       # creates a temporary local variable
    println("x + 1 = ", y)
    y               # show macro also returns the result
end

# assignments should create the variable
@show x = 3
let y = x = 2 
    println("x = 2 = ", y)
    y
end
x                   # should be equal to 2
```

The original Julia's [implementation](https://github.com/JuliaLang/julia/blob/ae8452a9e0b973991c30f27beb2201db1b0ea0d3/base/show.jl#L946-L959) is not dissimilar to the following macro definition:
```@example lab07_show
macro myshow(ex)
    quote
        println($(QuoteNode(ex)), " = ", repr(begin local value = $(esc(ex)) end))
        value
    end
end
```
Testing it gives us the expected behavior
```@repl lab07_show
@myshow xx = 1 + 1
xx                  # should be defined
```
In this "simple" example, we had to use the following concepts mentioned already in the [lecture](@ref macro_lecture):
- `QuoteNode(ex)` is used to wrap the expression inside another layer of quoting, such that when it is interpolated into `:()` it stays being a piece of code instead of the value it represents - [**TRUE QUOTING**](@ref lec7_quotation)
- `esc(ex)` is used in case that the expression contains an assignment, that has to be evaluated in the top level module `Main` (we are `esc`aping the local context) - [**ESCAPING**](@ref lec7_hygiene)
- `$(QuoteNode(ex))` and `$(esc(ex))` is used to evaluate an expression into another expression. [**INTERPOLATION**](@ref lec7_quotation)
- `local value = ` is used in order to return back the result after evaluation

Lastly, let's mention that we can use `@macroexpand` to see how the code is manipulated in the `@myshow` macro
```@repl lab07_show
@macroexpand @show x + 1
```

## Repeat macro
In the profiling/performance [labs](@ref perf_lab) we have sometimes needed to run some code multiple times in order to gather some samples and we have tediously written out simple for loops inside functions such as this
```julia
function run_polynomial(n, a, x)
    for _ in 1:n
        polynomial(a, x)
    end
end
```

We can remove this boilerplate code by creating a very simple macro that does this for us.
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Define macro `@repeat` that takes two arguments, first one being the number of times a code is to be run and the other being the actual code.
```julia
julia> @repeat 3 println("Hello!")
Hello!
Hello!
Hello!
```
Before defining the macro, it is recommended to write the code manipulation functionality into a helper function `_repeat`, which helps in organization and debugging of macros.
```julia
_repeat(3, :(println("Hello!"))) # testing "macro" without defining it
```

**HINTS**:
- use `$` interpolation into a for loop expression; for example given `ex = :(1+x)` we can interpolate it into another expression `:($ex + y)` -> `:(1 + x + y)`
- if unsure what gets interpolated use round brackets `:($(ex) + y)`
- macro is a function that *creates* code that does what we want

**BONUS**:
What happens if we call `@repeat 3 x = 2`? Is `x` defined?

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@repl lab07_repeat
macro repeat(n::Int, ex)
    return _repeat(n, ex)
end

function _repeat(n::Int, ex)
    :(for _ in 1:$n
        $ex
     end)
end

_repeat(3, :(println("Hello!")))
@repeat 3 println("Hello!")
```
Even if we had used escaping the expression `x = 2` won't get evaluated properly due to the induced scope of the for loop. In order to resolve this we would have to specially match that kind of expression and generate a proper syntax withing the for loop `global $ex`. However we may just warn the user in the docstring that the usage is disallowed. 

```@raw html
</p></details>
```
Note that this kind of repeat macro is also defined in the [`Flux.jl`](https://fluxml.ai/) machine learning framework, wherein it's called `@epochs` and is used for creating training [loop](https://fluxml.ai/Flux.jl/stable/training/training/#Datasets).

## [Polynomial macro](@id lab07_polymacro)
This is probably the last time we are rewriting the `polynomial` function, though not quite in the same way. We have seen in the last [lab](@ref introspection_lab), that some optimizations occur automatically, when the compiler can infer the length of the coefficient array, however with macros we can *generate* optimized code directly (not on the same level - we are essentially preparing already unrolled/inlined code).

Ideally we would like to write some macro `@poly` that takes a polynomial in a mathematical notation and spits out an anonymous function for its evaluation, where the loop is unrolled. 

*Example usage*:
```julia
p = @poly x 3x^2+2x^1+10x^0  # the first argument being the independent variable to match
p(2) # return the value
```

However in order to make this happen, let's first consider much simpler case of creating the same but without the need for parsing the polynomial as a whole and employ the fact that macro can have multiple arguments separated by spaces.

```julia
p = @poly 3 2 10
p(2)
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Create macro `@poly` that takes multiple arguments and creates an anonymous function that constructs the unrolled code. Instead of directly defining the macro inside the macro body, create helper function `_poly` with the same signature that can be reused outside of it.

Recall Horner's method polynomial evaluation from previous [labs](@ref horner):
```julia
function polynomial(a, x)
    accumulator = a[end] * one(x)
    for i in length(a)-1:-1:1
        accumulator = accumulator * x + a[i]
        #= accumulator = muladd(x, accumulator, a[i]) =# # equivalent
    end
    accumulator  
end
```

**HINTS**:
- you can use `muladd` function as replacement for `ac * x + a[i]`
- think of the `accumulator` variable as the mathematical expression that is incrementally built (try to write out the Horner's method[^1] to see it)
- you can nest expression arbitrarily
- the order of coefficients has different order than in previous labs (going from high powers of `x` last to them being first)
- use `evalpoly` to check the correctness
```julia
using Test
p = @poly 3 2 10
@test p(2) == evalpoly(2, [10,2,3]) # reversed coefficients
```

[^1]: Explanation of the Horner schema can be found on [https://en.wikipedia.org/wiki/Horner%27s\_method](https://en.wikipedia.org/wiki/Horner%27s_method).
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@repl lab07_poly
using InteractiveUtils #hide
macro poly(a...)
    return _poly(a...)
end

function _poly(a...)
    N = length(a)
    ex = :($(a[1]))
    for i in 2:N
        ex = :(muladd(x, $ex, $(a[i]))) # equivalent of :(x * $ex + $(a[i]))
    end
    :(x -> $ex)
end

p = @poly 3 2 10
p(2) == evalpoly(2, [10,2,3])
@code_lowered p(2) # can show the generated code
```

```@raw html
</p></details>
```

Moving on to the first/harder case, where we need to parse the mathematical expression.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Create macro `@poly` that takes two arguments first one being the independent variable and second one being the polynomial written in mathematical notation. As in the previous case this macro should define an anonymous function that constructs the unrolled code. 
```julia
julia> p = @poly x 3x^2+2x^1+10x^0  # the first argument being the independent variable to match
```

**HINTS**:
- though in general we should be prepared for some edge cases, assume that we are really strict with the syntax allowed (e.g. we really require spelling out x^0, even though it is mathematically equivalent to `1`)
- reuse the `_poly` function from the previous exercise
- use the `MacroTools.jl` to match/capture `a_*$v^(n_)`, where `v` is the symbol of independent variable, this is going to be useful in the following steps
    1. get maximal rank of the polynomial
    2. get coefficient for each power

!!! note "`MacroTools.jl`"
    Though not the most intuitive, [`MacroTools.jl`](https://fluxml.ai/MacroTools.jl/stable/) pkg help us with writing custom macros. We will use two utilities
    #### `@capture`
    This macro is used to match a pattern in a *single* expression and return values of particular spots. For example
    ```julia
    julia> using MacroTools
    julia> @capture(:[1, 2, 3, 4, 5, 6, 7], [1, a_, 3, b__, c_])
    true
    
    julia> a, b, c
    (2,[4,5,6],7)
    ```
    #### `postwalk`/`prewalk`
    In order to extend `@capture` to more complicated expression trees, we can used either `postwalk` or `prewalk` to walk the AST and match expression along the way. For example
    ```julia
    julia> using MacroTools: prewalk, postwalk
    julia> ex = quote
         x = f(y, g(z))
         return h(x)
       end
    
    julia> postwalk(ex) do x
            @capture(x, fun_(arg_)) && println("Function: ", fun, " with argument: ", arg)
            x
        end;
    Function: g with argument: z
    Function: h with argument: x
    ```
    Note that the `x` or the iteration is required, because by default postwalk/prewalk replaces currently read expression with the output of the body of `do` block.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```@example lab07_poly
using MacroTools
using MacroTools: postwalk, prewalk

macro poly(v::Symbol, p::Expr)
    a = Tuple(reverse(_get_coeffs(v, p)))
    return _poly(a...)
end

function _max_rank(v, p)
    mr = 0
    postwalk(p) do x
        if @capture(x, a_*$v^(n_))
            mr = max(mr, n)
        end
        x
    end
    mr
end

function _get_coeffs(v, p)
    N = _max_rank(v, p) + 1
    coefficients = zeros(N)
    postwalk(p) do x
        if @capture(x, a_*$v^(n_))
            coefficients[n+1] = a
        end
        x
    end
    coefficients
end
```
Let's test it.
```@repl lab07_poly
p = @poly x 3x^2+2x^1+10x^0
p(2) == evalpoly(2, [10,2,3])
@code_lowered p(2) # can show the generated code
```

```@raw html
</p></details>
```

## Ecosystem macros
There are at least two ways how we can make our life simpler when using our `Ecosystem` and `EcosystemCore` pkgs. Firstly, recall that in order to test our simulation we always had to write something like this:
```julia
function create_world()
    n_grass       = 500
    regrowth_time = 17.0

    n_sheep         = 100
    Δenergy_sheep   = 5.0
    sheep_reproduce = 0.5
    sheep_foodprob  = 0.4

    n_wolves       = 8
    Δenergy_wolf   = 17.0
    wolf_reproduce = 0.03
    wolf_foodprob  = 0.02

    gs = [Grass(id, regrowth_time) for id in 1:n_grass];
    ss = [Sheep(id, 2*Δenergy_sheep, Δenergy_sheep, sheep_reproduce, sheep_foodprob) for id in n_grass+1:n_grass+n_sheep];
    ws = [Wolf(id, 2*Δenergy_wolf, Δenergy_wolf, wolf_reproduce, wolf_foodprob) for id in n_grass+n_sheep+1:n_grass+n_sheep+n_wolves];
    World(vcat(gs, ss, ws))
end
world = create_world();
```
which includes the tedious process of defining the agent counts, their parameters and last but not least the unique id manipulation. As part of the [HW](@ref hw07) for this lecture you will be tasked to define a simple DSL, which can be used to define a world in a few lines.

Secondly, the definition of a new `Animal` or `Plant`, that did not have any special behavior currently requires quite a bit of repetitive code. For example defining a new plant type `Broccoli` goes as follows
```julia
abstract type Broccoli <: PlantSpecies end
Base.show(io::IO,::Type{Broccoli}) = print(io,"🥦")

EcosystemCore.eats(::Animal{Sheep},::Plant{Broccoli}) = true
```

and definition of a new animal like a `Rabbit` looks very similar
```julia
abstract type Rabbit <: AnimalSpecies end
Base.show(io::IO,::Type{Rabbit}) = print(io,"🐇")

EcosystemCore.eats(::Animal{Rabbit},p::Plant{Grass}) = size(p) > 0
EcosystemCore.eats(::Animal{Rabbit},p::Plant{Broccoli}) = size(p) > 0
```
In order to make this code "clearer" (depends on your preference) we will create two macros, which can be called at one place to construct all the relations.

### New Animal/Plant definition
Our goal is to be able to define new plants and animal species, while having a clear idea about their relations. For this we have proposed the following macros/syntax:
```julia
@species Plant Broccoli 🥦
@species Animal Rabbit 🐇
@eats Rabbit [Grass => 0.5, Broccoli => 1.0, Mushroom => -1.0]
```
Unfortunately the current version of `Ecosystem` and `EcosystemCore`, already contains some definitions of species such as `Sheep`, `Wolf` and `Mushroom`, which may collide with definitions during prototyping, therefore we have created a modified version of those pkgs, which will be provided in the lab.

!!! note "Testing relations"
    We can test the current definition with the following code that constructs "eating matrix"
    ```julia
    using Ecosystem
    using Ecosystem.EcosystemCore
    
    function eating_matrix()
        _init(ps::Type{<:PlantSpecies}) = ps(1, 10.0)
        _init(as::Type{<:AnimalSpecies}) = as(1, 10.0, 1.0, 0.8, 0.7)
        function _check(s1, s2)
            try
                if s1 !== s2
                    EcosystemCore.eats(_init(s1), _init(s2)) ? "✅" : "❌"
                else
                    return "❌"
                end
            catch e
                if e isa MethodError
                    return "❔"
                else
                    throw(e)
                end
            end
        end
    
        animal_species = subtypes(AnimalSpecies)
        plant_species = subtypes(PlantSpecies)
        species = vcat(animal_species, plant_species)
        em = [_check(s, ss) for (s,ss) in Iterators.product(animal_species, species)]
        string.(hcat(["🌍", animal_species...], vcat(permutedims(species), em)))
    end
    eating_matrix()
     🌍  🐑  🐺  🌿  🍄
     🐑  ❌  ❌  ✅  ✅
     🐺  ✅  ❌  ❌  ❌
    ```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Based on the following example syntax, 
```julia
@species Plant Broccoli 🥦
@species Animal Rabbit 🐇
```
write macro `@species` inside `Ecosystem` pkg, which defines the abstract type, its show function and exports the type. For example `@species Plant Broccoli 🥦` should generate code:
```julia
abstract type Broccoli <: PlantSpecies end
Base.show(io::IO,::Type{Broccoli}) = print(io,"🥦")
export Broccoli
```
Define first helper function `_species` to inspect the macro's output. This is indispensable, as we are defining new types/constants and thus we may otherwise encounter errors during repeated evaluation (though only if the type signature changed).
```julia
_species(:Plant, :Broccoli, :🥦)
_species(:Animal, :Rabbit, :🐇)
```

**HINTS**:
- use `QuoteNode` in the show function just like in the `@myshow` example
- escaping `esc` is needed for the returned in order to evaluate in the top most module (`Ecosystem`/`Main`)
- ideally these changes should be made inside the modified `Ecosystem` pkg provided in the lab (though not everything can be refreshed with `Revise`) - there is a file `ecosystem_macros.jl` just for this purpose
- multiple function definitions can be included into a `quote end` block
- interpolation works with any expression, e.g. `$(typ == :Animal ? AnimalSpecies : PlantSpecies)`

**BONUS**:
Based on `@species` define also macros `@animal` and `@plant` with two arguments instead of three, where the species type is implicitly carried in the macro's name.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
Macro `@species`
```julia
macro species(typ, name, icon)
    esc(_species(typ, name, icon))
end

function _species(typ, name, icon)
    quote
        abstract type $name <: $(typ == :Animal ? AnimalSpecies : PlantSpecies) end
        Base.show(io::IO, ::Type{$name}) = print(io, $(QuoteNode(icon)))
        export $name
    end
end

_species(:Plant, :Broccoli, :🥦)
_species(:Animal, :Rabbit, :🐇)
```

And the bonus macros `@plant` and `@animal`
```julia
macro plant(name, icon)
    return :(@species Plant $name $icon)
end

macro animal(name, icon)
    return :(@species Animal $name $icon)
end
```

```@raw html
</p></details>
```

The next exercise applies macros to the agents eating behavior.
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Define macro `@eats` inside `Ecosystem` pkg that assigns particular species their eating habits via `eat!` and `eats` functions. The macro should process the following example syntax
```julia
@eats Rabbit [Grass => 0.5, Broccoli => 1.0],
```
where `Grass => 0.5` defines the behavior of the `eat!` function. The coefficient is used here as a multiplier for the energy balance, in other words the `Rabbit` should get only `0.5` of energy for a piece of `Grass`.

**HINTS**:
- ideally these changes should be made inside the modified `Ecosystem` pkg provided in the lab (though not everything can be refreshed with `Revise`) - there is a file `ecosystem_macros.jl` just for this purpose
- escaping `esc` is needed for the returned in order to evaluate in the top most module (`Ecosystem`/`Main`)
- you can create an empty `quote end` block with `code = Expr(:block)` and push new expressions into its `args` incrementally
- use dispatch to create specific code for the different combinations of agents eating other agents (there may be catch in that we have to first `eval` the symbols before calling in order to know if they are animals or plants)

!!! note "Reminder of `EcosystemCore` `eat!` and `eats` functionality"
    In order to define that an `Wolf` eats `Sheep`, we have to define two methods
    ```
    EcosystemCore.eats(::Animal{Wolf}, ::Animal{Sheep}) = true
    
    function EcosystemCore.eat!(ae::Animal{Wolf}, af::Animal{Sheep}, w::World)
        incr_energy!(ae, $(multiplier)*energy(af)*Δenergy(ae))
        kill_agent!(af, w)
    end
    ```
    In order to define that an `Sheep` eats `Grass`, we have to define two methods
    ```
    EcosystemCore.eats(::Animal{Sheep}, p::Plant{Grass}) = size(p)>0

    function EcosystemCore.eat!(a::Animal{Sheep}, p::Plant{Grass}, w::World)
        incr_energy!(a, $(multiplier)*size(p)*Δenergy(a))
        p.size = 0
    end
    ```

**BONUS**:
You can try running the simulation with the newly added agents.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
macro eats(species::Symbol, foodlist::Expr)
    return esc(_eats(species, foodlist))
end


function _generate_eat(eater::Type{<:AnimalSpecies}, food::Type{<:PlantSpecies}, multiplier)
    quote
        EcosystemCore.eats(::Animal{$(eater)}, p::Plant{$(food)}) = size(p)>0
        function EcosystemCore.eat!(a::Animal{$(eater)}, p::Plant{$(food)}, w::World)
            incr_energy!(a, $(multiplier)*size(p)*Δenergy(a))
            p.size = 0
        end
    end
end

function _generate_eat(eater::Type{<:AnimalSpecies}, food::Type{<:AnimalSpecies}, multiplier)
    quote
        EcosystemCore.eats(::Animal{$(eater)}, ::Animal{$(food)}) = true
        function EcosystemCore.eat!(ae::Animal{$(eater)}, af::Animal{$(food)}, w::World)
            incr_energy!(ae, $(multiplier)*energy(af)*Δenergy(ae))
            kill_agent!(af, w)
        end
    end
end

_parse_eats(ex) = Dict(arg.args[2] => arg.args[3] for arg in ex.args if arg.head == :call && arg.args[1] == :(=>))

function _eats(species, foodlist)
    cfg = _parse_eats(foodlist)
    code = Expr(:block)
    for (k,v) in cfg
        push!(code.args, _generate_eat(eval(species), eval(k), v))
    end
    code
end

species = :Rabbit 
foodlist = :([Grass => 0.5, Broccoli => 1.0])
_eats(species, foodlist)
```

```@raw html
</p></details>
```

---
## Resources
- macros in Julia [documentation](https://docs.julialang.org/en/v1/manual/metaprogramming/#man-macros)

### `Type{T}` type selectors
We have used `::Type{T}` signature[^2] at few places in the `Ecosystem` family of packages (and it will be helpful in the HW as well), such as in the `show` methods
```julia
Base.show(io::IO,::Type{World}) = print(io,"🌍")
```
This particular example defines a method where the second argument is the `World` type itself and not an instance of a `World` type. As a result we are able to dispatch on specific types as values. 

Furthermore we can use subtyping operator to match all types in a hierarchy, e.g. `::Type{<:AnimalSpecies}` matches all animal species

[^2]: [https://docs.julialang.org/en/v1/manual/types/#man-typet-type](https://docs.julialang.org/en/v1/manual/types/#man-typet-type)
