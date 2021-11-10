# [Lab 07: Macros](@id macro_lab)
In this lab we are goinging to explore avenues of where macros can be useful
- convenience (`@repeat n code`, `@show`)
- code generation (`@define animal`)
- syntactic sugar (`@world`)
- performance critical applications (`@poly`)

## Show macro
Let's start with dissecting the simple `@show` macro, which allows us to demonstrate advanced concepts of macros
- true quoting
- escaping
- interpolation

```@repl lab07_show
x = 1
@show x + 1                 # equivalent to
let y = x + 1 
    println("x + 1 = ", y)
    y                # show macro also returns the result
end

@show x = 3
let y = x = 2 
    println("x = 2 = ", y)
    y
end
x
```
We have to both evaluate the code and show the expression as a string, which cannot be done easily in the realm of normal functions, however macro has programatic access to the code and thus also the name of the variable. We can use `@macroexpand` to see what is happening:
```@repl lab07_show
@macroexpand @show x + 1
```
Compared to the original [implementation](https://github.com/JuliaLang/julia/blob/ae8452a9e0b973991c30f27beb2201db1b0ea0d3/base/show.jl#L946-L959)
```julia
"""
    @show
Show an expression and result, returning the result. See also [`show`](@ref).
"""
macro show(exs...)
    blk = Expr(:block)
    for ex in exs
        push!(blk.args, :(println($(sprint(show_unquoted,ex)*" = "),
                                  repr(begin local value = $(esc(ex)) end))))
    end
    isempty(exs) || push!(blk.args, :value)
    return blk
end
```
Though this looks complicated we can boil it down in the following `@myshow` macro.
```julia
macro myshow(ex)
    :(println($(QuoteNode(ex)), " = ", repr(begin local value = $(esc(ex)) end)))
end

@myshow xx = 1 + 1
xx                  # should be defined
```
Notice the following:
- `QuoteNode(ex)` is used to wrap the expression inside another layer of quoting, such that when it is interpolated into `:()` it stays being a piece of code instead of the value it represents
- `esc(ex)` is used in case that the expression contains an assignment, that has to be evaluated in the top level module `Main` (we are `esc`aping the local context)
- `$(QuoteNode(ex))` and `$(esc(ex))` is used to evaluate an expression into another expression.
All of the macros here should be hygienic.

## Repeat macro
In the lecture on profiling we have sometimes needed to run some code multiple times in order to gather some samples and we have tediously written out simple for loops inside functions such as this
```julia
function run_polynomial(n, a, x)
    for _ in 1:n
        polynomial(a, x)
    end
end
```

We can simplify this by creating a macro that does this for us.
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

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
macro repeat(n::Int, ex)
    :(for _ in 1:$n
        $(esc(ex))
     end)
end
```

```@raw html
</p></details>
```

## [Polynomial macro](@id lab07_polymacro)
This is probably the last time we are rewritting the `polynomial` function, though not quite in the same way. We have seen in the last lab, that some optimizations occur automatically, when the compiler can infer the length of the coefficient array, however with macros we can generate code optimized code directly (not on the same level - we are essentially preparing already unrolled/inlined code).

Ideally we would like write some macro `@poly` that takes a polynomial in a mathematical notation and spits out an anonymous function for its evaluation, where the loop is unrolled. 

*Example usage*:
```julia
julia> p = @poly x 3x^2 + 2x^1 + 10x^0  # the first argument being the independent variable to match
```

However in order to make this happen, let's first consider much simpler case of creating the same but without the need for parsing the polynomial as a whole and employ the fact that macro can have multiple arguments separated by spaces.

```julia
julia> p = @poly 3 2 10
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Create macro `@poly` that takes multiple arguments and creates an anonymous function that constructs the unrolled code. Instead of directly defining the macro inside the macro body, create helper function `_poly` with the same signature that can be reused outside of it.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
macro poly(a...)
    return _poly(a...)
end

function _poly(a...)
    N = length(a)
    ex = :($(a[1]))
    for i in 2:N
        ex = :(muladd(x, $ex, $(a[i])))
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


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Create macro `@poly` that takes two arguments (see [above](@ref lab07_polymacro)) and creates an anonymous function that constructs the unrolled code. 

**HINTS**:
- though in general we should be prepared for some edge cases, assume that we are really strict with the syntax allowed
- reuse `_poly` that we have defined in the previous exercise
- use macro tools to match `a_*$v^(n_)`, where `v` is the symbol of independent variable
    + get maximal rank of the polynomial
    + get coefficient for each power

!!! info "`MacroTools.jl`"
    Though not the most intuitive, `MacroTools.jl` pkg allows us to play with macros.
    - `@capture`
    - `postwalk`/`prewalk`
    **TODO**

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
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

p = @poly x 3x^2 + 2x^1 + 10
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

EcosystemCore.eats(::Animal{Rabbit},::Plant{Grass}) = true
EcosystemCore.eats(::Animal{Rabbit},::Plant{Broccoli}) = true
```
In order to make these relation clearer we will create two macros, which can be called at one place to construct all the relations.

### New Animal/Plant definition
Our goal is to be able to define new plants and animal species, while having a clear idea about their relations. For this we have proposed the following macros/syntax:
```julia
@plant begin
    name  -> Broccoli 
    icon  -> 🥦
end

@animal begin
    name -> Rabbit 
    icon -> 🐇
    eats -> [Grass => 0.5ΔE, Broccoli => 1.0ΔE, Mushroom => -1.0ΔE]
end
```


Unfortunately the current version of `Ecosystem` and `EcosystemCore`, already contains some definitions of species such as `Sheep`, `Wolf` and `Mushroom`, which would collide with the new definition thus there exists a modified version of those pkgs. **TODO LINK IT**

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
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Define macros `@plant` and `@animal`, which define the functionality of agents based on the following sample syntax
```julia
@plant begin
    name  => Broccoli 
    icon  => 🥦
end

@animal begin
    name => Rabbit 
    icon => 🐇
    eats => [Grass => 0.5ΔE, Broccoli => 1.0ΔE, Mushroom => -1.0ΔE]
end
```
Syntax `Grass => 0.5ΔE` indicates defines the behavior of the `eat!` function, where the coefficient is used as a multiplier for the energy balance, in other words the `Rabbit` should get only `0.5` of energy for a piece of `Grass`.

Define first helper functions `_plant` and `_animal` to inspect the respective macro's output. This is indispensable, as we are defining new types/constants and thus we would otherwise encountered errors during repeated evaluation (though only if the type signature changed).

**HINTS**:
- use `QuoteNode` in the show function
- sdfsdf

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia

ex = :(begin
    name  => Broccoli 
    icon  => 🥦
end)

macro plant(ex)
    return _plant(ex)
end

function _plant(ex)
    cfg = Dict{Symbol, Symbol}()
    for arg in ex.args
        if ~(arg isa LineNumberNode) && arg.head == :call && arg.args[1] == :(=>)
            carg = arg.args
            key = carg[2]
            val = carg[3]
            cfg[key] = val
        end
    end

    println(cfg)

    quote
        abstract type $(cfg[:name]) <: PlantSpecies end
        Base.show(io::IO, ::Type{$(cfg[:name])}) = print(io,"$(QuoteNode($(cfg[:icon])))")
    end
end

_plant(ex)


ex = :(begin
    name => Rabbit 
    icon => 🐇
    eats => [Grass => 0.5, Broccoli => 1.0, Mushroom => -1.0]
end)

macro animal(ex)
    return _animal(ex)
end

_parse_eats(ex) = Dict(arg.args[2] => arg.args[3] for arg in ex.args if arg.head == :call && arg.args[1] == :(=>))
function _generate_eat(eater::Type{<:AnimalSpecies}, food::Type{<:PlantSpecies}, multiplier)
    quote
        EcosystemCore.eats(::Animal{$(eater)}, ::Plant{$(food)}) = true
        function EcosystemCore.eat!(a::Animal{$(eater)}, p::Plant{$(food)}, w::World)
            if size(p)>0
                incr_energy!(a, $(multiplier)*size(p)*Δenergy(a))
                p.size = 0
            end
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

function _animal(ex)
    cfg = Dict{Symbol, Any}()
    for arg in ex.args
        if ~(arg isa LineNumberNode) && arg.head == :call && arg.args[1] == :(=>)
            carg = arg.args
            key = carg[2]
            val = carg[3]
            cfg[key] = key == :eats ? _parse_eats(val) : val
        end
    end

    code = quote
        abstract type $(cfg[:name]) <: AnimalSpecies end
        Base.show(io::IO, ::Type{$(cfg[:name])}) = print(io,"$(QuoteNode($(cfg[:icon])))")
    end

    for (k,v) in cfg[:eats]
        push!(code.args, _generate_eat(cfg[:name], k, v)) # does not work without first defining the type tree
    end
    code
end

_animal(ex)
```

```@raw html
</p></details>
```

---
# Resources
- macros in Julia [documentation](https://docs.julialang.org/en/v1/manual/metaprogramming/#man-macros)