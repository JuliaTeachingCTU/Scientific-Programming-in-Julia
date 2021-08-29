# Motivation

Before going into details about Julia type system, we spent few minutes motivating the two main role of type system, which is (i) structuring the code and (ii) and communicating to the compiler your intentions how the type will be used. While the first aspect is very important for the convenience of programmer and possible abstraction in the language, the latter aspect is very important for speed. 

What Wikipedia tells about type and type system?

*In computer science and computer programming, a **data type** or simply **type** is an attribute of data which tells the compiler or interpreter how the programmer intends to use the data (see ![wiki](https://en.wikipedia.org/wiki/Data_type])).* 

*A **type system** is a logical system comprising a set of rules that assigns a property called a type to the various constructs of a computer program, such as variables, expressions, functions or modules. These types formalize and enforce the otherwise implicit categories the programmer uses for algebraic data types, data structures, or other components (see ![wiki](https://en.wikipedia.org/wiki/Type_system])).*

## Structuring the code
The main role is therefore aiding help to **structure** the code and impose semantic restriction.
Consider for example two types with the same definition but different names.
```julia
struct Dog
	name::String
end

struct Cat
	name::String
end
```
This allows us to define functions applicable only to the corresponding type
```julia
bark(dog::Dog) = println(dog.name, " has barked.")
meow(cat::Cat) = println(cat.name, " have meowed.")
```
and therefore the compiler (or interpretter) enforces that dog can only `bark` and never `meow` and vice versa can cat only `meow`. In this sense, it ensures that `bark(cat)` and `meow(dog)` never happen. Unlike if we define 
those functions as 
```julia
bark(dog::String) = println(dog.name, " has barked.")
meow(cat::String) = println(cat.name, " have meowed.")
```

## Intention of use and restrictions on compilers
The *intention of use* in types is tightly related to how efficient code can compiler produce for that given intention. As an example, consider a following two variables `a` and `b` and function `inc` which increments the content by one and return the array.
```julia
a = [1, 2, 3]
b = (1, 2, 3)
inc(x) = x .+ 1
```
The variable `a` is an array of `Int64`s, whereas `b` is a Tuple of `Int64`. Now if we look, how compiler compiles each version, you will see a striking difference
```julia
@code_native inc(a)
@code_native inc(b)
```
On my i5-8279U CPU, the difference is visible in performance
```julia
using BenchmarkTools
julia> @btime inc($(a));
  36.779 ns (1 allocation: 112 bytes)

julia> @btime inc($(b));
  0.036 ns (0 allocations: 0 bytes)
```
(as will be seen later, the difference in speed comes mainly from the memory allocation). 
For fun you can also test the speed of not type stable code
```julia
c = Vector{Number}([1, 2, 3])
julia> @btime inc($(c));
  865.304 ns (9 allocations: 464 bytes)
```

Does it mean that we should always use `Tuples` instead of `Arrays`? Surely not, it is just that each is better for different use-case. Arrays allows us for example to reuse space, which Tuples do not permit. For example
```julia
inc!(x) = x .= x .+ 1
```
will work for `a` but not for `b`, as Tuples are "immutable". This gives the compiler freedom to allocate (typically on Stack), while arrays are (at the time of writing) allocated strinctly on heap (needless to say that non-allocating version `inc!` of `inc` is much faster).

# The type system

## Julia is dynamicaly typed
Julia's type system is dynamic, which means that all types are resolved during runtime. But, if the compiler can infer type of all variables of a function, it can specialize it leading to a very efficient code. Consider again the above example
```
a = [1,2,3]
c = Vector{Number}([1,2,3])
inc!(x) = x .= x .+ 1
```
where in case of calling `inc(a)`, the compiler precisely knows types of items of `a` (all are `Int64`, which allow it to compile `inc(a)` for a vector if Integers. In case of `c`, the compiler does not know, what to expect, therefore he has to create a generic version of `inc` where for each item he has to decide the type. This way more complex leading to drop in performance
```julia
julia> @btime inc!($(a))
  4.322 ns (0 allocations: 0 bytes)
julia> @btime inc!($(c))
  132.294 ns (6 allocations: 96 bytes)  
```

## Types of types
Julia divides types into three classes.

### Abstract Type
(![Julia documentation](https://docs.julialang.org/en/v1/manual/types/#man-abstract-types))
Abstract types cannot be instantiated, which means that we cannot create a variable that would have an abstract type (try `typeof(Number(1f0))`). The most important use of abstract type is for structuring the code and defining general functions over semantically similar entities with different implementation. 

An abstract type is define by preceding a definition of a type (declared using `struct` keyword) with a keyword `abstract`. For example following set of abstract types defines the part number system in julia.
```julia
abstract type Number end
abstract type Real     <: Number end
abstract type AbstractFloat <: Real end
abstract type Integer  <: Real end
abstract type Signed   <: Integer end
abstract type Unsigned <: Integer end
```
The <: means "is a subtype of" and it is used in declarations where the right-hand is an immediate sypertype of a given type (`Integer` has an immediate supertype `Real`.) The abstract type `Number` is derived from `Any` which is a default supertype of any type (this means all subtypes are derived from `Any`).  

Recall the structure is used mainly for defining functions, that are known to provide a correct output on all subtypes of a given abstract type. Consider for example a `sgn` function, which can be defined as 
```julia
sgn(x::Real) = x > 0 ? 1 : x < 0 ? -1 : 0
sgn(x::Unsigned) = Int(x > 0)
```
and where the first function is defined for any subtype of a real, where the latter is used for subtypes of Unsigned.

```markdown
!!! info "Header"

How does the julia decides which function to use when two possible function definitions are possible? For example `sgn(UInt8(0))` can call a definition `sgn(x::Real)` or a definition `sgn(x::Unsigned)`. The answer is that it chooses the most specific version, and therefore for `sgn(UInt8(0))` it takes `sgn(x::Unsinged)`. If the compiler cannot decide, it throws an error.
```

A hierarchy of abstract types allows to define default implementation of some function over subtypes and specialize it for concrete types. A prime example is matrix multiplication, which has a generic implementation with many specializations for various types of matrices (sparse, dense, banded, etc.)

All abstract types, except `Any` are defined by core libraries. This means that Julia does not make a difference between abstract types that are shipped with the language and those defined by the user. All are treated the same.

Abstract types can have parameters, for we have for example 
```julia
abstract struct AbstractArray{T,N} end
```
which defines arrays of arbitrary dimentions with an element type `T`. Naturaly `AbstractArray{Float32,2}` is different from `AbstractArray{Float64,2}` and from `AbstractArray{Real,2}` (more on this later).

### Primitive types
Citing the ![documentation](https://docs.julialang.org/en/v1/manual/types/#Primitive-Types): *A primitive type is a concrete type whose data consists of plain old bits. Classic examples of primitive types are integers and floating-point values. Unlike most languages, Julia lets you declare your own primitive types, rather than providing only a fixed set of built-in ones. In fact, the standard primitive types are all defined in the language itself:*
```julia
primitive type Float16 <: AbstractFloat 16 end
primitive type Float32 <: AbstractFloat 32 end
primitive type Float64 <: AbstractFloat 64 end
```
We refer the reader to the original documentation, as we will not use them. They are mentioned to assure the reader that there is very little in Julia of what is not defined in it.

### Composite types
The composite types are similar to `struct` in C (they even have the same memory layout). It is not a great idea to think about them as objects (in OOP sense), because objects tie together *data* and *functions* over the data. Contrary in Julia (and in C), the function operates over data, but are not tied to them. Composite types are a powerhorse of Julia's type system, as most user-defined types are composite.

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

### Type hierarchy

Let's now





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

### Optional stuff

### Alignment in the memory
* Super-importnat for speed, cache utilization
* Know if you are row or column major 
* Vector{Vector{Float}} is a really bad idea over 

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
