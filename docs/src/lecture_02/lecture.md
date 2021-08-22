# Motivation

Before going into details about Julia type system, we spent few minutes motivating the two main role of type system, which is (i) structuring the code and (ii) and communicating to the compiler your intentions how the type will be used. While the first aspect is very important for the convenience of programmer and possible abstraction in the language, the latter aspect is very important for speed. 

What Wikipedia tells about type and type system?

*In computer science and computer programming, a **data type** or simply **type** is an attribute of data which tells the compiler or interpreter how the programmer intends to use the data (see ![wiki](https://en.wikipedia.org/wiki/Data_type])).* 

*A **type system** is a logical system comprising a set of rules that assigns a property called a type to the various constructs of a computer program, such as variables, expressions, functions or modules.[1] These types formalize and enforce the otherwise implicit categories the programmer uses for algebraic data types, data structures, or other components (see ![wiki](https://en.wikipedia.org/wiki/Type_system])).*

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
```
bark(dog::Dog) = println(dog.name, " has barked.")
meow(cat::Cat) = println(cat.name, " have meowed.")
```
and therefore the compiler (or interpretter) enforces that dog can only `bark` and never `meow` and vice versa can cat only `meow`. In this sense, it ensures that `bark(cat)` and `meow(dog)` never happen. Unlike if we define 
those functions as 
```
bark(dog::String) = println(dog.name, " has barked.")
meow(cat::String) = println(cat.name, " have meowed.")
```

## Intention of use and restrictions on compilers
The *intention of use* in types is tightly related to how efficient code can compiler produce for that given intention. As an example, consider a following two variables `a` and `b` and function `inc` which increments the content by one and return the array.
```
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

Does it mean that we should always use `Tuples` instead of `Arrays`, it is just that each is better for different use-case. Arrays allows us to reuse the space, for example
```julia
inc!(x) = x .= x .+ 1
```
will work for `a` but not for `b`, as Tuples are "immutable". This gives the compiler freedom to allocate them where he wishes (typically on Stack), while arrays are (at the time of writing) allocated strinctly on heap (needless to say that non-allocating version `inc!` of `inc` is much faster).

# The type system



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
```
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
