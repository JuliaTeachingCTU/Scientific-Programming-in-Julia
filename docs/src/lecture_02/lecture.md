# [Motivation](@id type_lecture)
Before going into the details of Julia's type system, we will spend a few minutes motivating the roles of a type system, which are:

1. Structuring code
2. Communicating to the compiler how a type will be used

The first aspect is important for the convenience of the programmer and enables abstractions
in the language, the latter aspect is important for the speed of the generated code. *Writing efficient Julia code is best viewed as a dialogue between the programmer and the compiler.* [^1] 


Type systems according to [Wikipedia](https://en.wikipedia.org/wiki/Data_type):
* In computer science and computer programming, a **data type** or simply **type** is an attribute of data which tells the compiler or interpreter how the programmer intends to use the data.
* A **type system** is a logical system comprising a set of rules that assigns a property called a type to the various constructs of a computer program, such as variables, expressions, functions or modules. These types formalize and enforce the otherwise implicit categories the programmer uses for algebraic data types, data structures, or other components.

## Structuring the code / enforcing the categories
The role of **structuring** the code and imposing semantic restriction
means that the type system allows you to logically divide your program,
and to prevent certain types of errors.
Consider for example two types, `Wolf` and `Sheep` which share the same
definition but the types have different names.

````@example lecture
struct Wolf
  name::String
  energy::Int
end

struct Sheep
  name::String
  energy::Int
end
````


This allows us to define functions applicable only to the corresponding type

````@example lecture
howl(wolf::Wolf) = println(wolf.name, " has howled.")
baa(sheep::Sheep) = println(sheep.name, " has baaed.")
nothing # hide
````

Therefore the compiler (or interpreter) **enforces** that a wolf can only `howl`
and never `baa` and vice versa a sheep can only `baa`. In this sense, it ensures
that `howl(sheep)` and `baa(wolf)` never happen.
```
baa(Sheep("Karl",3))
baa(Wolf("Karl",3))
```
Notice the type of error of the latter call `baa(Wolf("Karl",3))`. Julia raises `MethodError` which states that it has failed to find a function `baa` for the type `Wolf` (but there is a function `baa` for type `Sheep`).

For comparison, consider an alternative definition which does not have specified types

````@example lecture
bark(animal) = println(animal.name, " has howled.")
baa(animal)  = println(animal.name, " has baaed.")
nothing # hide
````
in which case the burden of ensuring that a wolf will never baa rests upon the
programmer which inevitably leads to errors (note that severely constrained
type systems are difficult to use).

## Intention of use and restrictions on compilers
Types play an important role in generating efficient code by a compiler, because they tells the compiler which operations are permitted, prohibited, and can indicate invariants of type (e.g. constant size of an array). If compiler knows that something is invariant (constant), it can expoit such information. As an example, consider the following two
alternatives to represent a set of animals:

````@example lecture
a = [Wolf("1", 1), Wolf("2", 2), Sheep("3", 3)]
b = (Wolf("1", 1), Wolf("2", 2), Sheep("3", 3))
nothing # hide
````

where `a` is an array which can contain arbitrary types and have arbitrary length
whereas `b` is a `Tuple` which has fixed length in which the first two items are of type `Wolf`
and the third item is of type `Sheep`. Moreover, consider a function which calculates the
energy of all animals as

````@example lecture
energy(animals) = mapreduce(x -> x.energy, +, animals)
nothing # hide
````

A good compiler makes use of the information provided by the type system to generate efficient code which we can verify by inspecting the compiled code using `@code_native` macro

````@repl lecture; ansicolor=true
@code_native debuginfo=:none energy(a)
@code_native debuginfo=:none energy(b)
````

one observes the second version produces more optimal code. Why is that?
* In the first representation, `a`, the animals are stored in an `Array{Any}` which can have arbitrary size and can contain arbitrary animals. This means that the compiler has to compile `energy(a)` such that it works on such arrays.
* In the second representation, `b`, the animals are stored in a `Tuple`, which specializes for lengths and types of items. This means that the compiler knows the number of animals and the type of each animal on each position within the tuple, which allows it to specialize.

This difference will indeed have an impact on the time of code execution.
On my i5-8279U CPU, the difference (as measured by BenchmarkTools) is

```julia
using BenchmarkTools
@btime energy($(a))
@btime energy($(b))
```
```
  70.2 ns (0 allocations: 0 bytes)
  2.62 ns (0 allocations: 0 bytes)
```

Which nicely demonstrates that the choice of types affects performance. Does it mean that we should always use `Tuples` instead of `Arrays`? Surely not, it is  just that each is better for different use-cases. Using Tuples means that the compiler will compile a special function for each length of tuple and each combination of types of items it contains, which is clearly wasteful.

# [Julia's type system](@id type_system)

## Julia is dynamicaly typed
Julia's type system is dynamic, which means that all types are resolved during runtime. **But**, if the compiler can infer types of all variables of the called function, it can specialize the function for that given type of variables which leads to efficient code. Consider a modified example where we represent two wolfpacks:

````@example lecture
wolfpack_a =  [Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
wolfpack_b =  Any[Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
nothing # hide
````

`wolfpack_a` carries a type `Vector{Wolf}` while `wolfpack_b` has the type `Vector{Any}`. This means that in the first case, the compiler knows that all items are of the type `Wolf`and it can specialize functions using this information. In case of `wolfpack_b`, it does not know which animal it will encounter (although all are of the same type), and therefore it needs to dynamically resolve the type of each item upon its use. This ultimately leads to less performant code.

```julia
@benchmark energy($(wolfpack_a))
@benchmark energy($(wolfpack_b))
```
```
  3.7 ns (0 allocations: 0 bytes)
  69.4 ns (0 allocations: 0 bytes)
```

To conclude, julia is indeed a dynamically typed language, **but** if the compiler can infer all types in a called function in advance, it does not have to perform the type resolution during execution, which produces performant code. This means and in hot (performance critical) parts of the code, you should be type stable, in other parts, it is not such big deal.

## Classes of types
Julia divides types into three classes: primitive, composite, and abstract.

### Primitive types
Citing the [documentation](https://docs.julialang.org/en/v1/manual/types/#Primitive-Types):  *A primitive type is a concrete type whose data consists of plain old bits. Classic examples  of primitive types are integers and floating-point values. Unlike most languages, Julia lets you declare your own primitive types, rather than providing only a fixed set of built-in ones. In fact, the standard primitive types are all defined in the language itself.*

The definition of primitive types look as follows
```julia
primitive type Float16 <: AbstractFloat 16 end
primitive type Float32 <: AbstractFloat 32 end
primitive type Float64 <: AbstractFloat 64 end
```
and they are mainly used to jump-start julia's type system. It is rarely needed to
define a special primitive type, as it makes sense only if you define special functions
operating on its bits. This is almost excusively used for exposing special operations
provided by the underlying CPU / LLVM compiler. For example `+` for `Int32` is different
from `+` for `Float32` as they call a different intrinsic operations. You can inspect this
jump-starting of the type system yourself by looking at Julia's source.
```julia
julia> @which +(1,2)
+(x::T, y::T) where T<:Union{Int128, Int16, Int32, Int64, Int8, UInt128, UInt16, UInt32, UInt64, UInt8} in Base at int.jl:87
```

At `int.jl:87`
```julia
(+)(x::T, y::T) where {T<:BitInteger} = add_int(x, y)
```
we see that `+` of integers is calling the function `add_int(x, y)`, which is defined in the core
part of the compiler in `Intrinsics.cpp` (yes, in C++), exposed in `Core.Intrinsics`

From Julia docs: *Core is the module that contains all identifiers considered "built in" to
the language, i.e. part of the core language and not libraries. Every module implicitly
specifies using Core, since you can't do anything without those definitions.*

Primitive types are rarely used, and they will not be used in this course. We mention them for the sake of completeness and refer the reader to the official Documentation (and source code of Julia).

An example of use of primitive type is a definition of one-hot vector in the library `PrimitiveOneHot` as 
```julia
primitive type OneHot{K} <: AbstractOneHotArray{1} 32 end
```
where `K` is the dimension of the one-hot vector. 

### Abstract types

An abstract type can be viewed as a set of concrete types. For example, an
`AbstractFloat` represents the set of concrete types `(BigFloat,Float64,Float32,Float16)`.
This is used mainly to define general methods for sets of types for which we expect the same behavior (recall the Julia design motivation: *if it quacks like a duck, waddles like a duck and looks like a duck, chances are it's a duck*). Abstract types are defined with `abstract type TypeName end`. For example the following set of abstract types defines part of julia's number system.
```julia
abstract type Number end
abstract type Real          <: Number end
abstract type Complex       <: Number end
abstract type AbstractFloat <: Real end
abstract type Integer       <: Real end
abstract type Signed        <: Integer end
abstract type Unsigned      <: Integer end
```
where `<:` means "is a subtype of" and it is used in declarations where the right-hand is an immediate sypertype of a given type (`Integer` has the immediate supertype `Real`.) If the supertype is not supplied, it is considered to be Any, therefore in the above defition `Number` has the supertype `Any`. 

We can list childrens of an abstract type using function `subtypes`  
```@example lecture
using InteractiveUtils: subtypes  # hide
subtypes(AbstractFloat)
```
and we can also list the immediate `supertype` or climb the ladder all the way to `Any` using `supertypes`
```@example lecture
using InteractiveUtils: supertypes  # hide
supertypes(AbstractFloat)
```

`supertype` and `subtypes` print only types defined in Modules that are currently loaded to your workspace. For example with Julia without any Modules, `subtypes(Number)` returns `[Complex, Real]`, whereas if I load `Mods` package implementing numbers defined over finite field, the same call returns `[Complex, Real, AbstractMod]`.

It is relatively simple to print a complete type hierarchy of 
````@example lecture
using AbstractTrees
function AbstractTrees.children(t::Type)
    t === Function ? Vector{Type}() : filter!(x -> x !== Any,subtypes(t))
end
AbstractTrees.printnode(io::IO,t::Type) = print(io,t)
print_tree(Number)
````

The main role of abstract types allows is in function definitions. They allow to define functions that can be used on variables with types with a given abstract type as a supertype. For example we can define a `sgn` function for **all** real numbers as

````@example lecture
sgn(x::Real) = x > 0 ? 1 : x < 0 ? -1 : 0
nothing # hide
````

and we know it would be correct for all real numbers. This means that if anyone creates
a new subtype of `Real`, the above function can be used. This also means that
**it is expected** that comparison operations are defined for any real number. Also notice that
`Complex` numbers are excluded, since they do not have a total order.

For unsigned numbers, the `sgn` can be simplified, as it is sufficient to verify if they are different (greater) than zero, therefore the function can read

````@example lecture
sgn(x::Unsigned) = x > 0 ? 1 : 0
nothing # hide
````

and again, it applies to all numbers derived from `Unsigned`. Recall that
`Unsigned <: Integer <: Real,` how does Julia decide,
which version of the function `sgn` to use for `UInt8(0)`? It chooses the most
specific version, and thus for `sgn(UInt8(0))` it will use `sgn(x::Unsinged)`.
If the compiler cannot decide, typically it encounters an ambiguity, it throws an error
and recommends which function you should define to resolve it.

The above behavior allows to define default "fallback" implementations and while allowing
to specialize for sub-types. A great example is matrix multiplication, which has a
generic (and slow) implementation with many specializations, which can take advantage
of structure (sparse, banded), or use optimized implementations (e.g. blas implementation
for dense matrices with eltype `Float32` and `Float64`).

Again, Julia does not make a difference between abstract types defined in `Base`
libraries shipped with the language and those defined by you (the user). All are treated
the same.

[From Julia documentation](https://docs.julialang.org/en/v1/manual/types/#man-abstract-types):
Abstract types cannot be instantiated, which means that we cannot create a variable that
would have an abstract type (try `typeof(Number(1f0))`). Also, abstract types cannot have
any fields, therefore there is no composition (there are lengthy discussions of why this is so,
one of the most definite arguments of creators is that abstract types with fields frequently lead
to children types not using some fields (consider circle vs. ellipse)).

### [Composite types](@id composite_types)
Composite types are similar to `struct` in C (they even have the same memory layout) as they logically join together other types. It is not a great idea to think about them as objects (in OOP sense), because objects tie together *data* and *functions* on owned data. Contrary in Julia (as in C), functions operate on data of structures, but are not tied to them and they are defined outside them. Composite types are workhorses of Julia's type system, as user-defined types are mostly composite (or abstract).

Composite types are defined using `struct TypeName [fields] end`. To define a position of an animal on the Euclidean plane as a type, we would write

````@example lecture
struct PositionF64
  x::Float64
  y::Float64
end
````

which defines a structure with two fields `x` and `y` of type `Float64`. Julia's compiler creates a default constructor, where both (but generally all) arguments are converted using `(convert(Float64, x), convert(Float64, y)` to the correct type. This means that we can construct a PositionF64 with numbers of different type that are convertable to Float64, e.g. `PositionF64(1,1//2)` but we cannot construct `PositionF64` where the fields would be of different type (e.g. `Int`, `Float32`, etc.) or they are not trivially convertable (e.g. `String`).

Fields in composite types do not have to have a specified type.  We can define a `VaguePosition` without specifying the type

````@example lecture
struct VaguePosition
  x
  y
end
````

This works as the definition above except that the arguments are not converted to `Float64` now. One can store different values in `x` and `y`, for example `String` (e.g. VaguePosition("Hello","world")). Although the above definition might be convenient, it limits the compiler's ability to specialize, as the type  `VaguePosition` does not carry information about type of `x` and `y`, which has a negative impact on the performance. For example

```julia
using BenchmarkTools
move(a,b) = typeof(a)(a.x+b.x, a.y+b.y)
x = [PositionF64(rand(), rand()) for _ in 1:100]
y = [VaguePosition(rand(), rand()) for _ in 1:100]
@benchmark reduce(move, $(x))
@benchmark reduce(move, $(y))
nothing # hide
```

Giving fields of a composite type an abstract type does not really solve the problem of the compiler not knowing the type. In this example, it still does not know, if it should use instructions for `Float64` or `Int8`.

```@example lecture
struct LessVaguePosition
  x::Real
  y::Real
end

z = [LessVaguePosition(rand(), rand()) for _ in 1:100];
@benchmark reduce(move, $(z))
nothing #hide
```

From the perspective of generating optimal code, both definitions are equally uninformative to the compiler as it cannot assume anything about the code. However, the  `LessVaguePosition` will ensure that the position will contain only numbers, hence catching trivial errors like instantiating `VaguePosition` with non-numeric types for which arithmetic operators will not be defined (recall the discussion on the  beginning of the lecture).

All structs defined above are immutable (as we have seen above in the case of `Tuple`), which means that one cannot change a field (unless the struct wraps a container, like and array, which allows that). For example this raises an error

````@repl lecture
a = LessVaguePosition(1,2)
a.x = 2
````

If one needs to make a struct mutable, use the keyword `mutable` before the keyword `struct` as

````@example lecture
mutable struct MutablePosition
  x::Float64
  y::Float64
end
````

In mutable structures, we can change the values of fields.

````@repl lecture
a = MutablePosition(1e0, 2e0)
a.x = 2;
a
````

Note, that the memory layout of mutable structures is different, as fields now contain references to memory locations, where the actual values are stored (such structures cannot be allocated on stack, which increases the pressure on Garbage Collector).

The difference can be seen from 
```
a, b = PositionF64(1,2), PositionF64(1,2)
@code_native debuginfo=:none move(a,b)
a, b = MutablePosition(1,2), MutablePosition(1,2)
@code_native debuginfo=:none move(a,b)
```
Why there is just one addition?

Also, the mutability is costly.
```julia
x = [PositionF43(rand(), rand()) for _ in 1:100];
z = [MutablePosition(rand(), rand()) for _ in 1:100];
@benchmark reduce(move, $(x))
@benchmark reduce(move, $(z))
```

### Parametric types
So far, we had to trade-off flexibility for generality in type definitions. Can we have both? The answer is affirmative. The way to achieve this  **flexibility** in definitions of the type while being  able to generate optimal code is to  **parametrize** the type definition. This is achieved by replacing types with a parameter (typically a single uppercase character) and decorating in definition by specifying different type in curly brackets. For example

```julia
struct PositionT{T}
  x::T
  y::T
end
u = [PositionT(rand(), rand()) for _ in 1:100]
u = [PositionT(rand(Float32), rand(Float32)) for _ in 1:100]

@benchmark reduce(move, $(u))
nothing # hide
```

Notice that the compiler can take advantage of specializing for different types (which does not have an effect here as in modern processors addition of `Float` and `Int` takes the same time).

```julia
v = [PositionT(rand(1:100), rand(1:100)) for _ in 1:100]
@benchmark reduce(move, v)
nothing #hide
```

The above definition suffers the same problem as `VaguePosition`, which is that it allows us to instantiate the `PositionT` with non-numeric types, e.g. `String`. We solve this by restricting the types `T` to be children of some supertype, in this case `Real`

````@example lecture
struct Position{T<:Real}
  x::T
  y::T
end
````

which will throw an error if we try to initialize it with `Position("1.0", "2.0")`. Notice the flexibility we have achieved. We can use `Position` to store (and later compute) not only over `Float32` / `Float64` but any real numbers defined by other packages, for example with `Posit`s.
```@example lecture
using SoftPosit
Position(Posit8(3), Posit8(1))
```
also notice that trying to construct the `Position` with different type of real numbers will fail, example `Position(1f0,1e0)`

Naturally, fields in structures can be of different types, as is in the below pointless example.

```@example lecture
struct PositionXY{X<:Real, Y<:Real}
  x::X
  y::Y
end
```

The type can be parametrized by a concrete types. This is usefuyl to communicate the compiler some useful informations, for example size of arrays. 

```julia
struct PositionZ{T<:Real,Z}
  x::T
  y::T
end

PositionZ{Int64,1}(1,2)
```


### Abstract parametric types
Like Composite types, Abstract types can also have parameters. These parameters define types that are common for all child types. A very good example is Julia's definition of arrays of arbitrary dimension `N` and type `T` of its items as
```julia
abstract type AbstractArray{T,N} end
```
Different `T` and `N` give rise to different variants of `AbstractArrays`,
therefore `AbstractArray{Float32,2}` is different from `AbstractArray{Float64,2}`
and from `AbstractArray{Float64,1}.` Note that these are still `Abstract` types,
which means you cannot instantiate them. Their purpose is
* to allow to define operations for broad class of concrete types
* to inform the compiler about constant values, which can be used
Notice in the above example that parameters of types do not have to be types, but can also be values of primitive types, as in the above example of `AbstractArray` `N` is the number of dimensions which is an integer value.

For convenience, it is common to give some important partially instantiated Abstract types an **alias**, for example `AbstractVector` as
```julia
const AbstractVector{T} = AbstractArray{T,1}
```
is defined in `array.jl:23` (in Julia 1.6.2), which allows us to define for example general prescription for the `dot` product of two abstract vectors as

````@example lecture
function dot(a::AbstractVector, b::AbstractVector)
  @assert length(a) == length(b)
  mapreduce(*, +, a, b)
end
nothing # hide
````

You can verify that the above general function can be compiled to performant code if
specialized for particular arguments.

````@example lecture; ansicolor=true
using InteractiveUtils: @code_native
@code_native debuginfo=:none mapreduce(*,+, [1,2,3], [1,2,3])
````

## More on the use of types in function definitions
### Terminology
A *function* refers to a set of "methods" for a different combination of type parameters (the term function can be therefore considered as refering to a mere **name**). *Methods* define different behavior for different types of arguments for a given function. For example

````@example lecture
move(a::Position, b::Position) = Position(a.x + b.x, a.y + b.y)
move(a::Vector{<:Position}, b::Vector{<:Position}) = move.(a,b)
nothing # hide
````

`move` refers to a function with methods `move(a::Position, b::Position)` and `move(a::Vector{<:Position}, b::Vector{<:Position})`. When different behavior on different types is defined by a programmer, as shown above, it is also called *implementation specialization*. There is another type of specialization, called *compiler specialization*, which occurs when the compiler generates different functions for you from a single method. For example for

````@repl lecture
move(Position(1,1), Position(2,2))
move(Position(1.0,1.0), Position(2.0,2.0))
````

the compiler generates two methods, one for `Position{Int64}` and the other for `Position{Float64}`. Notice that inside generated functions, the compiler needs to use different intrinsic operations, which can be viewed from

````@example lecture; ansicolor=true
@code_native debuginfo=:none move(Position(1,1), Position(2,2))
````

and

````@example lecture; ansicolor=true
@code_native debuginfo=:none move(Position(1.0,1.0), Position(2.0,2.0))
````

Notice that `move` works on `Posits` defined in 3rd party libas well
```
move(Position(Posit8(1),Posit8(1)), Position(Posit8(2),Posit8(2)))
```

## Intermezzo: How does the Julia compiler work?
Let's walk through an example. Consider the following definitions

````@example lecture
move(a::Position, by::Position) = Position(a.x + by.x, a.y + by.y)
move(a::T, by::T) where {T<:Position} = Position(a.x + by.x, a.y + by.y)
move(a::Position{Float64}, by::Position{Float64}) = Position(a.x + by.x, a.y + by.y)
move(a::Vector{<:Position}, by::Vector{<:Position}) = move.(a, by)
move(a::Vector{<:Position}, by::Position) = move.(a, by)
nothing # hide
````

and a function call

````@example lecture
a = Position(1.0, 1.0)
by = Position(2.0, 2.0)
move(a, by)
````

1. The compiler knows that you call the function `move`.
2. The compiler infers the type of the arguments. You can view the result with

````@repl lecture
(typeof(a),typeof(by))
````

3. The compiler identifies all `move`-methods with arguments of type `(Position{Float64}, Position{Float64})`:

````@repl lecture
wc = Base.get_world_counter()
m = Base.method_instances(move, (typeof(a), typeof(by)), wc)
m = first(m)
````

4a. If the method has been specialized (compiled), then the arguments are prepared and the method is invoked. The compiled specialization can be seen from

````@repl lecture
m.cache
````

4b. If the method has not been specialized (compiled), the method is compiled for the given type of arguments and continues as in step 4a.
A compiled function is therefore  a "blob" of **native code** living in a particular memory location. When Julia calls a function, it needs to pick the right block corresponding to a function with particular type of parameters.

If the compiler cannot narrow the types of arguments to concrete types, it has to perform the above procedure inside the called function, which has negative effects on performance, as the type resulution and identification of the methods can be slow, especially for methods with many arguments (e.g. 30ns for a method with one argument,
100 ns for method with two arguements). **You always want to avoid run-time resolution inside the performant loop!!!**
Recall the above example

```julia
wolfpack_a =  [Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
@benchmark energy($(wolfpack_a))
```
```
BenchmarkTools.Trial: 10000 samples with 991 evaluations.
 Range (min … max):  40.195 ns … 66.641 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     40.742 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   40.824 ns ±  1.025 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

   ▂▃ ▃▅▆▅▆█▅▅▃▂▂                                             ▂
  ▇██████████████▇▇▅▅▁▅▄▁▅▁▄▄▃▄▅▄▅▃▅▃▅▁▃▁▄▄▃▁▁▅▃▃▄▃▄▃▄▆▆▇▇▇▇█ █
  40.2 ns      Histogram: log(frequency) by time      43.7 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
```

and

```julia
wolfpack_b =  Any[Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
@benchmark energy($(wolfpack_b))
```
```
BenchmarkTools.Trial: 10000 samples with 800 evaluations.
 Range (min … max):  156.406 ns … 212.344 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     157.136 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   158.114 ns ±   4.023 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▅█▆▅▄▂   ▃▂▁                                                  ▂
  ██████▆▇██████▇▆▇█▇▆▆▅▅▅▅▅▃▄▄▅▄▄▄▄▅▁▃▄▄▃▃▄▃▃▃▄▄▄▅▅▅▅▁▅▄▃▅▄▄▅▅ █
  156 ns        Histogram: log(frequency) by time        183 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
```

An interesting intermediate between fully abstract and fully concrete type happens, when the compiler knows that arguments have abstract type, which is composed of a small number of concrete types. This case  called Union-Splitting, which happens when there is just a little bit of uncertainty. Julia will do something like
```julia
argtypes = typeof(args)
push!(execution_stack, args)
if T == Tuple{Int, Bool}
  @goto compiled_blob_1234
else # the only other option is Tuple{Float64, Bool}
  @goto compiled_blob_1236
end
```
For example

```julia
const WolfOrSheep = Union{Wolf, Sheep}
wolfpack_c =  WolfOrSheep[Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
@benchmark energy($(wolfpack_c))
```
```
BenchmarkTools.Trial: 10000 samples with 991 evaluations.
 Range (min … max):  43.600 ns … 73.494 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     44.106 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   44.279 ns ±  0.931 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

       █     ▁ ▃                                               
  ▂▂▂▆▃██▅▃▄▄█▅█▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▂▂▂▂▂ ▃
  43.6 ns         Histogram: frequency by time        47.4 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
```

Thanks to union splitting, Julia is able to have performant operations on arrays with undefined / missing values for example

````@repl lecture
[1, 2, 3, missing] |> typeof
````

### More on matching methods and arguments
In the above process, the step, where Julia looks for a method instance with corresponding parameters can be very confusing. The rest of this lecture will focus on this. For those who want to have a formal background, we recommend [talk of  Francesco Zappa Nardelli](https://www.youtube.com/watch?v=Y95fAipREHQ) and / or the one of [Jan Vitek](https://www.youtube.com/watch?v=LT4AP7CUMAw).

When Julia needs to specialize a method instance, it needs to find it among multiple definitions. A single function can have many method instances, see for example `methods(+)` which  lists all method instances of the `+`-function. How does Julia select the proper one?
1. It finds all methods where the type of arguments match or are subtypes of restrictions on arguments in the method definition.
2a. If there are multiple matches, the compiler selects the most specific definition.

2b. If the compiler cannot decide, which method instance to choose, it throws an error.

````@repl lecture
confused_move(a::Position{Float64}, by) = Position(a.x + by.x, a.y + by.y)
confused_move(a, by::Position{Float64}) = Position(a.x + by.x, a.y + by.y)
confused_move(Position(1.0,2.0), Position(1.0,2.0))
````

2c. If it cannot find a suitable method, it throws an error.

````@repl lecture
move(Position(1,2), VaguePosition("hello","world"))
````

Some examples: Consider following definitions

````@example lecture
move(a::Position, by::Position) = Position(a.x + by.x, a.y + by.y)
move(a::T, by::T) where {T<:Position} = T(a.x + by.x, a.y + by.y)
move(a::Position{Float64}, by::Position{Float64}) = Position(a.x + by.x, a.y + by.y)
move(a::Vector{<:Position}, by::Vector{<:Position}) = move.(a, by)
move(a::Vector{T}, by::Vector{T}) where {T<:Position} = move.(a, by)
move(a::Vector{<:Position}, by::Position) = move.(a, by)
nothing # hide
````

Which method will compiler select for

````@repl lecture
move(Position(1.0,2.0), Position(1.0,2.0))
````

The first three methods match the types of argumens, but the compiler will select the third one, since it is the most specific.

Which method will compiler select for

````@repl lecture
move(Position(1,2), Position(1,2))
````

Again, the first and second method definitions match the argument, but the second is the most specific.

Which method will the compiler select for

````@repl lecture
move([Position(1,2)], [Position(1,2)])
````

Again, the fourth and fifth method definitions match the argument, but the fifth is the most specific.

````@repl lecture
move([Position(1,2), Position(1.0,2.0)], [Position(1,2), Position(1.0,2.0)])
````

### Frequent problems
1. Why does the following fail?

````@repl lecture
foo(a::Vector{Real}) = println("Vector{Real}")
foo([1.0,2,3])
````

Julia's type system is **invariant**, which means that `Vector{Real}` is different from `Vector{Float64}` and from `Vector{Float32}`, even though `Float64` and `Float32` are sub-types of `Real`. Therefore `typeof([1.0,2,3])` isa `Vector{Float64}` which is not subtype of `Vector{Real}.` For **covariant** languages, this would be true. For more information on variance in computer languages, [see here](https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)). If the above definition of `foo` should be applicable to all vectors which has elements of subtype of `Real` we have define it as

````@example lecture
foo(a::Vector{T}) where {T<:Real} = println("Vector{T} where {T<:Real}")
nothing # hide
````

or equivalently but more tersely as

````@example lecture
foo(a::Vector{<:Real}) = println("Vector{T} where {T<:Real}")
nothing # hide
````

2. Diagonal rule says that a repeated type in a method signature has to be a concrete type (this is to avoid ambinguity if the repeated type is used inside function definition to define a new variable to change type of variables). Consider for example the function below

````@example lecture
move(a::T, b::T) where {T<:Position} = T(a.x + by.x, a.y + by.y)
nothing # hide
````

we cannot call it with `move(Position(1.0,2.0), Position(1,2))`, since in this case `Position(1.0,2.0)` is of type `Position{Float64}` while `Position(1,2)` is of type `Position{Int64}`.
3. When debugging why arguments do not match a particular method definition, it is useful to use `typeof`, `isa`, and `<:` commands. For example

````@repl lecture
typeof(Position(1.0,2.0))
typeof(Position(1,2))
Position(1,2) isa Position{Float64}
Position(1,2) isa Position{Real}
Position(1,2) isa Position{<:Real}
typeof(Position(1,2)) <: Position{<:Float64}
typeof(Position(1,2)) <: Position{<:Real}
````

### A bizzare definition which you can encounter
The following definition of a one-hot matrix is taken from [Flux.jl](https://github.com/FluxML/Flux.jl/blob/1a0b51938b9a3d679c6950eece214cd18108395f/src/onehot.jl#L10-L12)

````@example lecture
struct OneHotArray{T<:Integer, L, N, var"N+1", I<:Union{T,AbstractArray{T, N}}} <: AbstractArray{Bool, var"N+1"}
  indices::I
end
````

The parameters of the type carry information about the type used to encode the position of `one` in each column in `T`, the dimension of one-hot vectors in `L`, the dimension of the storage of `indices` in `N` (which is zero for `OneHotVector` and one for `OneHotMatrix`), number of dimensions of the `OneHotArray` in `var"N+1"` and the type of underlying storage of indicies `I`.


[^1]: Type Stability in Julia, Pelenitsyn et al., 2021](https://arxiv.org/pdf/2109.01950.pdf)
