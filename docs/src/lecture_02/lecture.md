# Motivation

Before going into details about Julia type system, we spent few minutes motivating the two main role of type system, which is (i) structuring the code and (ii) and communicating to the compiler your intentions how the type will be used. While the first aspect is very important for the convenience of programmer and possible abstraction in the language, the latter aspect is very important for speed. 

What Wikipedia tells about type and type system?

*In computer science and computer programming, a **data type** or simply **type** is an attribute of data which tells the compiler or interpreter how the programmer intends to use the data (see ![wiki](https://en.wikipedia.org/wiki/Data_type])).* 

*A **type system** is a logical system comprising a set of rules that assigns a property called a type to the various constructs of a computer program, such as variables, expressions, functions or modules. These types formalize and enforce the otherwise implicit categories the programmer uses for algebraic data types, data structures, or other components (see ![wiki](https://en.wikipedia.org/wiki/Type_system])).*

## Structuring the code
The main role is therefore aiding help to **structure** the code and impose semantic restriction.
Consider for example two types with the same definition but different names.
```julia
struct Wolf
	name::String
  energy::Int
end

struct Sheep
	name::String
  energy::Int
end
```
This allows us to define functions applicable only to the corresponding type
```julia
howl(wolf::Wolf) = println(wolf.name, " has howled.")
baa(sheep::Sheep) = println(sheep.name, " has baaed.")
```
and therefore the compiler (or interpretter) enforces that wolf can only `howl` and never `baa` and vice versa sheep can only `baa`. In this sense, it ensures that `howl(sheep)` and `baa(wolf)` never happen. Unlike if we define 
those functions as 
```julia
bark(animal) = println(animal.name, " has howled.")
baa(animal)  = println(animal.name, " has baaed.")
```

## Intention of use and restrictions on compilers
The *intention of use* in types is related to how efficient code can compiler produce for that given intention. As an example, consider a following two alternatives to represent a set of animals:
```julia
a = [Wolf("1", 1), Wolf("2", 2), Sheep("3", 3)]
b = (Wolf("1", 1), Wolf("2", 2), Sheep("3", 3))
```
and define a function to sum energy of all animals as 
```
energy(animals) = mapreduce(x -> x.energy, +, animals)
```
Inspecting the compiled code using 
```julia
@code_native energy(a)
@code_native energy(b)
```
one observes the second version produces more optimal code. Why is that?

* In the first representation, `a`, animals are stored in `Array` which can have arbitrary size and can contain arbitrary animals. This means that compiler has to compile `energy(a)` such that it works on such arrays.

* In the second representation, `b`, animals are stored in `Tuple`, which specializes for lengths and types of items. This means that the compiler knows the number of animals and the type of each animal on each position within the tuple, which allows him to specialize.

This difference will be indeed measurable
On my i5-8279U CPU, the difference is visible in performance
```julia
using BenchmarkTools
julia> @btime energy(a);
  86.944 ns (0 allocations: 0 bytes)

julia> @btime energy(b);
  16.206 ns (0 allocations: 0 bytes)
```

Which nicely demonstrates that smart choice of types can greatly affect the performance.

Does it mean that we should always use `Tuples` instead of `Arrays`? Surely not, it is just that each is better for different use-case. Using Tuples means that compiler will compile special function for each tuple it observes, which is clearly wasteful.

# The type system

## Julia is dynamicaly typed
Julia's type system is dynamic, which means that all types are resolved during runtime. But, if the compiler can infer type of all variables of a function, it can specialize it leading to a very efficient code. Consider a slightly modified example where we represent two wolfpacks:
```
wolfpack_a =  [Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
wolfpack_b =  Any[Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
```
`wolfpack_a` carries a type `Vector{Wolf}` while `wolfpack_b` has a type `Vector{Any}`. This means that in the first case, the compiler know that all items are of the type `Wolf` and it can specialize functions using this information. In case of `wolfpack_b`, he does not know which animal he will encounter (although all are of the same type), and therefore it needs to dynamically resolve the type of each item upon its use. This ultimately leads to less performant code.
```julia
julia> @btime energy(wolfpack_a)
  15.498 ns (0 allocations: 0 bytes)
julia> @btime energy(wolfpack_b)
  91.152 ns (0 allocations: 0 bytes)
```

To conclude, julia is indeed dynamically typed language, **but** if the compiler can infer all types in called function in advance, it does not have to perform a type resolution during execution, which produces a performant code.

## Types of types
Julia divides types into three classes: primitive, composite, and abstract.

### Primitive types
Citing the ![documentation](https://docs.julialang.org/en/v1/manual/types/#Primitive-Types): *A primitive type is a concrete type whose data consists of plain old bits. Classic examples of primitive types are integers and floating-point values. Unlike most languages, Julia lets you declare your own primitive types, rather than providing only a fixed set of built-in ones. In fact, the standard primitive types are all defined in the language itself:*

The definition of primitive types look as follows
```julia
primitive type Float16 <: AbstractFloat 16 end
primitive type Float32 <: AbstractFloat 32 end
primitive type Float64 <: AbstractFloat 64 end
```
and they are mainly used to jump-start julia's type system. It rarely make a sense to define a special primitive type, as it make sense only if you define special functions operating on its bits, which makes mostly sense if you want to expose special operations provided by underlying CPU / LLVM compiler. For example `+` for `Int32` is different from `+` for `Float32` as they call a different intrinsic operation. You can inspect this jump-starting of type system by yourself by inspecting Julia's source.
```julia
julia> @which +(1,2)
+(x::T, y::T) where T<:Union{Int128, Int16, Int32, Int64, Int8, UInt128, UInt16, UInt32, UInt64, UInt8} in Base at int.jl:87
```

At `int.jl:87` 
```julia
(+)(x::T, y::T) where {T<:BitInteger} = add_int(x, y)
```
we seen that `+` of integers is calling function `add_int(x, y)`, which is defined in a core part of the compiler in `Intrinsics.cpp` (yes, in C++).

From Julia docs: *Core is the module that contains all identifiers considered "built in" to the language, i.e. part of the core language and not libraries. Every module implicitly specifies using Core, since you can't do anything without those definitions.*

Primitive types are rarely used, and they will not be used in this course. We touch them for the sake of completness and refer reader to the official Documentation (and source code of Julia).

### [Composite types](@id composite_types)
The composite types are similar to `struct` in C (they even have the same memory layout). It is not a great idea to think about them as objects (in OOP sense), because objects tie together *data* and *functions* over owned data. Contrary in Julia (as in C), the function operates over data, but are not tied to them. Composite types are a workhorses of Julia's type system, as most user-defined types are composite.

The composite type is defined as
```julia
struct Position
  x::Float64
  y::Float64
end
```
which defines a structure with two fields `x` and `y` of type `Float64`. Julia compiler creates a default constructor, where both (but generally all) arguments are converted using `(convert(Float64, x), convert(Float64, y)` to the correct type. This means that we can construct a Position with numbers of different type that are convertable to Float, e.g. `Position(1,1//2)`.

Composite types do not have to have a specified type, e.g.
```julia
struct VaguePosition
  x 
  y 
end
```
which would work as the definition above and allows to store different values in `x`, for example `String`. But it would limit compiler's ability to specialize, which can have a negative impact on the performance. For example
```julia
using BenchmarkTools
move(a::T, b::T) where {T} = T(a.x + b.x, a.y + b.y)
x = [Position(rand(), rand()) for _ in 1:100]
y = [VaguePosition(rand(), rand()) for _ in 1:100]
julia> @btime reduce(move, x);
  114.105 ns (1 allocation: 32 bytes)

julia> @btime reduce(move, y);
  3.879 μs (199 allocations: 3.12 KiB)
```

The same holds if the Composite type contains field with AbstractType, for example
```julia
struct LessVaguePosition
  x::Real
  y::Real 
end
z = [LessVaguePosition(rand(), rand()) for _ in 1:100];
julia> @btime reduce(move, z);
  16.260 μs (496 allocations: 9.31 KiB)
```

A recommended way to fix this is to parametrize the struct is to parametrize the type definition as follows
```julia
struct Position2{T}
  x::T
  y::T 
end
u = [Position2(rand(), rand()) for _ in 1:100];
julia> @btime reduce(move, u);
  110.043 ns (1 allocation: 32 bytes)
```
and notice that the compiler can take advantage of specializing for differenty types (which does not have effect as in modern processrs have addition of Float and Int takes the same time)
```julia
v = [Position2(rand(1:100), rand(1:100)) for _ in 1:100];
@btime reduce(move, v);
  110.043 ns (1 allocation: 32 bytes)
```

All structs defined above are immutable (as we have seen above in the case of `Tuple`), which means that one cannot change a field (unless the struct wraps a container, like and array, which allows that). For example this raises an error
```julia
a = Position(1, 2)
a.x = 2
```

If one needs to make a struct mutable, use the keyword `mutable` as follows
```julia
mutable struct MutablePosition{T}
  x::T
  y::T

end

a = MutablePosition(1e0, 2e0)
a.x = 2
```

but there might be some performance penalty(not observable at this simple demo).


### Abstract Type 
(![Julia documentation](https://docs.julialang.org/en/v1/manual/types/#man-abstract-types))
Abstract types cannot be instantiated, which means that we cannot create a variable that would have an abstract type (try `typeof(Number(1f0))`). Also, Abstract types cannot have any fields (therefore there is no composition).
The most important use of abstract type is for structuring the code and defining general functions over semantically similar entities with different implementation. 

An abstract types are defined by preceding a definition of a type (declared using `struct` keyword) with a keyword `abstract`. For example following set of abstract types defines part of julia's number systems.
```julia
abstract type Number end
abstract type Real     <: Number end
abstract type AbstractFloat <: Real end
abstract type Integer  <: Real end
abstract type Signed   <: Integer end
abstract type Unsigned <: Integer end
```
The `<:` means "is a subtype of" and it is used in declarations where the right-hand is an immediate sypertype of a given type (`Integer` has an immediate supertype `Real`.) The abstract type `Number` is derived from `Any` which is a default supertype of any type (this means all subtypes are derived from `Any`).  

The type hiearchy is used for defining general functions, that are known to provide a correct output on all subtypes of a given abstract type. For example a `sgn` function can be defined for **all** real numbers as 
```julia
sgn(x::Real) = x > 0 ? 1 : x < 0 ? -1 : 0
```
and we know it would be correct for all real numbers. This means that if anyone creates a new subtype of `Real`, the above function can be used.

For unsigned numbers, the `sgn` can be simplified, as it is sufficient to verify if they are different (greated) then zeros, therefore the function can read
```julia
sgn(x::Unsigned) = Int(x > 0)
```
and again, it applies to all numbers derived from `Unsigned`. Recall that `Unsigned <: Integer <: Real,` how does Julia decides, which version of the function `sgn` to use for `UInt8(0)`? It chooses the most specific version, and therefore for `sgn(UInt8(0))` it will use `sgn(x::Unsinged)`. If the compiler cannot decide, typically it encounters an ambiguity, it throws an error and recommend which function you should define to resolve it.

The above behavior allows to define default "fallback" implementations and while allowing to specialize for sub-types. A usual example is a matrix multiplication, which has a generic (and slow) implementation with many specializations, which can take advantage of structure (sparse, banded), or of optimized implementations (e.g. blas implementation for dense matrices with eltype `Float32` and `Float64`).

Again, Julia does not make a difference between abstract types defined in `Base` libraries shipped with the language and those defined by you (the user). All are treated the same.


Like Composite types, Abstract types can have parameters. For example Julia defines an array of arbitrary dimension `N` and type `T` of its items as
```julia
abstract type AbstractArray{T,N} end
```
Different `T` and `N` gives rise to different variants of `AbstractArrays`, therefore `AbstractArray{Float32,2}` is different from `AbstractArray{Float64,2}` and from `AbstractArray{Float64,1}.` Note that these are still `Abstract` types, which means you cannot instantiate them. They purpose is
* to allow to define operations for broad class of concrete types
* to inform compiler about constant values, which can be used 

For convenience, you can name some important partially instantiated Abstract types, for example `AbstractVector` as 
```
const AbstractVector{T} = AbstractArray{T,1}
```
is defined in `array.jl:23` (in Julia 1.6.2), which allows us to define for example general prescription for `dot` product of two abstract vectors as 
```julia
function dot(a::AbstractVector, b::AbstractVector)
  @assert length(a) == length(b)
  mapreduce(*, +, a, b)
end
```
You can verify that the above general function can be compiled to a performant code if specialized for a particular arguments.
```julia
@code_native mapreduce(*,+, [1,2,3], [1,2,3])
```
.

### More on Type hierarchy and Parametric types
How types and functions comes together and what is the role of their parametrisation?




An interesting feature of Julia's type system is parametrisation of types, which we have slightly touched above. `Abstract` and `Composite` types can be parametrised, where the parameter can be other type, sets of types, or or a value of any bits type.

Let's look at some examples. Julia defines an abstract type of arrays of various sizes and shapes as 
```
abstract type AbstractArray{T,N} end
```
where it is expected `T` to be a type of the element of arrays and and `N` the number of dimenctions. **It is expected** that every type derived from an abstract array will implement certain function (there is an expected *interface*)





Depending on the variance of the type constructor, the subtyping relation of the simple types may be either preserved, reversed, or ignored for the respective complex types. In the OCaml programming language, for example, "list of Cat" is a subtype of "list of Animal" because the list type constructor is covariant. This means that the subtyping relation of the simple types are preserved for the complex types. 

On the other hand, "function from Animal to String" is a subtype of "function from Cat to String" because the function type constructor is contravariant in the parameter type. Here the subtyping relation of the simple types is reversed for the complex types. 


### The power of Type System \& multiple dispatch
  - Zero cost abstraction the compiler has information about types and can freely inline
    `x |> identity |> identity |> identity |> identity |> identity`

```julia
julia> f(x) = x |> identity |> identity |> identity |> identity |> identity
@code_lowered f(1)
@code_lowered f(1)
```
  - Why the type system is important for efficiency
  - Bad practices 
  - **LABS:**
      + Number type-tree
      + Design Interval Aritmetics (for ODES)

### VS NOtes: 
Basics:
 - type hierarchy
 - subtyping
 - Unions


## Pevnak's idea 
- Type hierarchy and rationale behind
	* Why I cannot create an abstract struct with fields.
	* Why a type cannot subtype more than one types
	* How the type matching system works and what are the rules (Would take Jan Vitek's lecture or earlier) https://youtu.be/Y95fAipREHQ?t=958c


- What types do in practice and how it matters?
	* Allow to structure the program (example two types with the same memory layout (even empty) specializes methods)
	* Provides an information how to arrange variables in computer memory
	* Inform compiler how things can be stored (possibly on stack (bit types) vs strictly on heap (arrays))
	* Inform compiler about what is known (mutable vs non-mutable structs). Explain that when things are mutable, they have to be "boxed", meaning the variable is on stack and the structure contains pointer. 
	* The effect of not strictly typed structs, where the type inference of objects is left to runtime

- Performance gotchas
	- Why global is slow and how `const` comes to rescue

- Show the above with @code_typed, @code_native and @btime the effects, such that they can see that.



- Stefans's C++ example of overloading https://discourse.julialang.org/t/claim-false-julia-isnt-multiple-dispatch-but-overloading/42370/16
_ Discussion about multiple inheritance https://github.com/JuliaLang/julia/issues/5


### Examples

* why `Vector{AbstractFloat}` is a bad idea, while `Vector{Float64}` is a good one?
* why `Vector{AbstractFloat}` is different to `Vector{<:AbstractFloat}` or `Vector{T} where {T<:AbstractFloat}` 


### Why [1,2,3] is not a Vector{Number}

Vector{Number} is a concrete type, and [1, 2, 3] has type Vector{Int}, which is also a concrete type. One concrete type is never a subtype of another concrete type, they are the leaves of the type tree.

Vector{Number} is concrete, even though Number is not a concrete type. That’s because it has a concrete implementation which can store all types that are subtypes of Number, it has a specific memory layout etc. On the other hand, AbstractVector{Int} is the other way around and not a concrete type, because the container is abstract even though the element is concrete.

What you can do instead is [1, 2, 3] isa Vector{<:Number} which is true. That’s because <:Number is a sort of placeholder which means “any type which is a subtype of Number”. This is often needed for dispatching on containers where you want to allow set of element types. f(x::Vector{Number}) can take only arguments of type Vector{Number}, whereas g(x::Vector{<:Number}) can take e.g. Vector{Int}, Vector{Float64}, Vector{Real}, Vector{Number}, etc.

### Frequently asked (and discussed) questions

* Why Abstract type cannot have a fields. In OOP, abstract classes can define fields that would be common to all derived classes. 

* Why type cannot be derived from multiple abstract type (mimicking Multiple inheritance). It seems like the function resolution might be difficult. An example copied from discussion in a 5th issue of the language opened in 2011 (and still not closed) ![](https://github.com/JuliaLang/julia/issues/5) show this usecase
```julia
abstract type A end;
abstract tyle B end;

f(::A) = 1
f(::B) = 2

struct C <: A,B end;  # read as multiple inheritance.
```
with that, it is not clear which function the compiler should call when `f(C())`. A current consensus seems to favour trait system support on the language level then the multiple inheritance, but priorities are elsewhere at the moment.

### A Headache examples
* This is a great example for type resolution.
```julia
function Base.reduce(::typeof(hcat), xs::Vector{TV})  where {T, L, TV<:OneHotLike{T, L}}
  OneHotMatrix(reduce(vcat, map(_indices, xs)), L)
end
```


### Turning multiple-distpatch to single dispatch

https://stackoverflow.com/questions/39133424/how-to-create-a-single-dispatch-object-oriented-class-in-julia-that-behaves-l/39150509#39150509


### conversion among variables

reinterpret(Float32, ref_sketch)

### Trivia 
why this is true Vector{Int} <: AbstractVector{<:Any}
why this is false Vector{Int} <: Vector{Any}
why this is true Vector{Int} <: Vector{<:Any}
