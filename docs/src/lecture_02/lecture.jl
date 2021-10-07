# # Motivation

using InteractiveUtils # hide
using InteractiveUtils: subtypes # hide

# Before going into details about Julia type system, we will spend a few minutes motivating 
# the two main roles of the type system, which are:
# 
# 1. Structuring the code
# 2. Communicating to the compiler how your type will be used.
# 
# The first aspect is important for the convenience of the programmer and enables abstractions
# in the language, the latter aspect is important for the speed of the generated code. 
# 
# Type systems according to [Wikipedia](https://en.wikipedia.org/wiki/Data_type):
# * In computer science and computer programming, a **data type** or simply **type** is an attribute of data which tells the compiler or interpreter how the programmer intends to use the data.* 
# * A **type system** is a logical system comprising a set of rules that assigns a property called a type to the various constructs of a computer program, such as variables, expressions, functions or modules. These types formalize and enforce the otherwise implicit categories the programmer uses for algebraic data types, data structures, or other components.*
# 
# ## Structuring the code / enforcing the categories
# The role of **structuring** the code and imposing semantic restriction
# means that the type system allows you to logically divide your program, 
# and to prevent certain types of errors.
# Consider for example two types, `Wolf` and `Sheep` which share the same 
# definition but the types have different names.

struct Wolf
  name::String
  energy::Int
end

struct Sheep
  name::String
  energy::Int
end

# This allows us to define functions applicable only to the corresponding type
# 
howl(wolf::Wolf) = println(wolf.name, " has howled.")
baa(sheep::Sheep) = println(sheep.name, " has baaed.")
nothing # hide
# 
# Therefore the compiler (or interpretter) **enforces** that a wolf can only `howl`
# and never `baa` and vice versa a sheep can only `baa`. In this sense, it ensures 
# that `howl(sheep)` and `baa(wolf)` never happen.
# For comparison, consider an alternative definition as follows
# 
bark(animal) = println(animal.name, " has howled.")
baa(animal)  = println(animal.name, " has baaed.")
# 
# in which case the burden of ensuring that a wolf will never baa rests upon the
# programmer which inevitably leads to errors (note that severely constrained 
# type systems are difficult to use).

# ## Intention of use and restrictions on compilers
# The *intention of use* in types is related to how efficient code a compiler can
# produce for that given intention. As an example, consider a following two 
# alternatives to represent a set of animals:
a = [Wolf("1", 1), Wolf("2", 2), Sheep("3", 3)]
b = (Wolf("1", 1), Wolf("2", 2), Sheep("3", 3))
# where `a` is an array which can contain arbitrary types and have arbitrary length
# whereas `b` is a `Tuple` which has fixed length in which the first two items are of type `Wolf`
# and the third item is of type `Sheep`. Moreover, consider a function which calculates the
# energy of all animals as 
energy(animals) = mapreduce(x -> x.energy, +, animals)
# A good compiler makes use of the information provided by the type system to generate effiecint code
# which we can verify by inspecting the compiled code using `@code_native` macro
@code_native energy(a)
#
@code_native energy(b)
# one observes the second version produces more optimal code. Why is that?
# * In the first representation, `a`, the animals are stored in an `Array` which can have arbitrary size and can contain arbitrary animals. This means that the compiler has to compile `energy(a)` such that it works on such arrays.
# * In the second representation, `b`, the animals are stored in a `Tuple`, which specializes for lengths and types of items. This means that the compiler knows the number of animals and the type of each animal on each position within the tuple, which allows it to specialize.

# This difference will indeed have an impact on the time of code execution. 
# On my i5-8279U CPU, the difference (as measured by BenchmarkTools) is
using BenchmarkTools
@btime energy(a);
@btime energy(b);
# Which nicely demonstrates that the choice of types affects performance. Does it mean that we should always use `Tuples` instead of `Arrays`? Surely not, it is  just that each is better for different use-cases. Using Tuples means that the compiler will compile a special function for each length of tuple and each combination of types of items it contains, which is clearly wasteful.

# # Julia's type system

# ## Julia is dynamicaly typed
# Julia's type system is dynamic, which means that all types are resolved during runtime. **But**, if the compiler can infer types of all variables of the called function, it can specialize the function for that given type of variables which leads to efficient code. Consider a modified example where we represent two wolfpacks:
wolfpack_a =  [Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
wolfpack_b =  Any[Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
# `wolfpack_a` carries a type `Vector{Wolf}` while `wolfpack_b` has the type `Vector{Any}`. This means that in the first case, the compiler knows that all items are of the type `Wolf`and it can specialize functions using this information. In case of `wolfpack_b`, it does not know which animal it will encounter (although all are of the same type), and therefore it needs to dynamically resolve the type of each item upon its use. This ultimately leads to less performant code.
@btime energy(wolfpack_a)
@btime energy(wolfpack_b)
nothing # hide

# To conclude, julia is indeed a dynamically typed language, **but** if the compiler can infer 
# all types in a called function in advance, it does not have to perform the type resolution 
# during execution, which produces performant code.

# ## Classes of types
# Julia divides types into three classes: primitive, composite, and abstract.

# ### Primitive types
# Citing the [documentation](https://docs.julialang.org/en/v1/manual/types/#Primitive-Types):  *A primitive type is a concrete type whose data consists of plain old bits. Classic examples  of primitive types are integers and floating-point values. Unlike most languages, Julia lets you declare your own primitive types, rather than providing only a fixed set of built-in ones. In fact, the standard primitive types are all defined in the language itself.*

# The definition of primitive types look as follows
# ```julia
# primitive type Float16 <: AbstractFloat 16 end
# primitive type Float32 <: AbstractFloat 32 end
# primitive type Float64 <: AbstractFloat 64 end
# ```
# and they are mainly used to jump-start julia's type system. It is rarely needed to 
# define a special primitive type, as it makes sense only if you define special functions 
# operating on its bits. This is almost excusively used for exposing special operations 
# provided by the underlying CPU / LLVM compiler. For example `+` for `Int32` is different 
# from `+` for `Float32` as they call a different intrinsic operations. You can inspect this 
# jump-starting of the type system yourself by looking at Julia's source.
# ```julia
# julia> @which +(1,2)
# +(x::T, y::T) where T<:Union{Int128, Int16, Int32, Int64, Int8, UInt128, UInt16, UInt32, UInt64, UInt8} in Base at int.jl:87
# ```

# At `int.jl:87` 
# ```julia
# (+)(x::T, y::T) where {T<:BitInteger} = add_int(x, y)
# ```
# we see that `+` of integers is calling the function `add_int(x, y)`, which is defined in the core 
# part of the compiler in `Intrinsics.cpp` (yes, in C++).

# From Julia docs: *Core is the module that contains all identifiers considered "built in" to 
# the language, i.e. part of the core language and not libraries. Every module implicitly 
# specifies using Core, since you can't do anything without those definitions.*

# Primitive types are rarely used, and they will not be used in this course. We mention them 
# for the sake of completeness and refer the reader to the official Documentation (and source code 
# of Julia).


# ### Abstract Type 
# 
# An abstract type can be viewed as a set of concrete types. For example, an
# `AbstractFloat` represents the set of concrete types `(BigFloat,Float64,Float32,Float16)`.
# This is used mainly to define general methods for sets of types for which we expect the same behavior (recall the Julia design motivation: *if it quacks like a duck, waddles like a duck and looks like a duck, chances are it's a duck*). Abstract types are defined with `abstract type TypeName end`. For example the following set of abstract types defines part of julia's number system.
# ```julia
# abstract type Number end
# abstract type Real          <: Number end
# abstract type Complex       <: Number end
# abstract type AbstractFloat <: Real end
# abstract type Integer       <: Real end
# abstract type Signed        <: Integer end
# abstract type Unsigned      <: Integer end
# ```
# where `<:` means "is a subtype of" and it is used in declarations where the right-hand is an immediate sypertype of a given type (`Integer` has the immediate supertype `Real`.) If the supertype is not supplied, it is considered to be Any, therefore in the above defition `Number` has the supertype `Any`. Children of a particular type can be viewed as 
using AbstractTrees
function AbstractTrees.children(t::Type)
    t === Function ? Vector{Type}() : filter!(x -> x !== Any,subtypes(t))
end
AbstractTrees.printnode(io::IO,t::Type) = print(io,t)
print_tree(Number)
# As was mentioned, abstract types allows as to define functions that can be applied variebles of types with a given abstract type as a supertype. For example we can define a `sgn` function for **all** real numbers as 
sgn(x::Real) = x > 0 ? 1 : x < 0 ? -1 : 0
nothing # hide
# and we know it would be correct for all real numbers. This means that if anyone creates 
# a new subtype of `Real`, the above function can be used. This also means that 
# **it is expected** that comparison operations are defined for any real number. Also notice that 
# `Complex` numbers are excluded, since they do not have a total order.

# For unsigned numbers, the `sgn` can be simplified, as it is sufficient to verify if they are different (greater) than zero, therefore the function can read
sgn(x::Unsigned) = x > 0 ? 1 : 0
nothing # hide
# and again, it applies to all numbers derived from `Unsigned`. Recall that 
# `Unsigned <: Integer <: Real,` how does Julia decide, 
# which version of the function `sgn` to use for `UInt8(0)`? It chooses the most 
# specific version, and thus for `sgn(UInt8(0))` it will use `sgn(x::Unsinged)`. 
# If the compiler cannot decide, typically it encounters an ambiguity, it throws an error 
# and recommends which function you should define to resolve it.

# The above behavior allows to define default "fallback" implementations and while allowing 
# to specialize for sub-types. A great example is matrix multiplication, which has a 
# generic (and slow) implementation with many specializations, which can take advantage 
# of structure (sparse, banded), or use optimized implementations (e.g. blas implementation 
# for dense matrices with eltype `Float32` and `Float64`).

# Again, Julia does not make a difference between abstract types defined in `Base` 
# libraries shipped with the language and those defined by you (the user). All are treated 
# the same.

# (![From Julia documentation](https://docs.julialang.org/en/v1/manual/types/#man-abstract-types))
# Abstract types cannot be instantiated, which means that we cannot create a variable that 
# would have an abstract type (try `typeof(Number(1f0))`). Also, abstract types cannot have 
# any fields, therefore there is no composition (there are lengty discussions of why this is so,
# one of the most definite arguments of creators is that abstract types with fields frequently lead
# to children types not using some fields (consider circle vs. ellipse)).

# ### [Composite types](@id composite_types)
# Composite types are similar to `struct` in C (they even have the same memory layout) as they logically join together other types. It is not a great idea to think about them as objects (in OOP sense), because objects tie together *data* and *functions* on owned data. Contrary in Julia (as in C), functions operate on data of structures, but are not tied to them and they are defined outside them. Composite types are workhorses of Julia's type system, as user-defined types are mostly composite (or abstract).

# Composite types are defined using `struct TypeName [fields] end`. To define a position of an animal on the Euclidean plane as a type, we would write
struct PositionF64
  x::Float64
  y::Float64
end
# which defines a structure with two fields `x` and `y` of type `Float64`. Julia's compiler creates a default constructor, where both (but generally all) arguments are converted using `(convert(Float64, x), convert(Float64, y)` to the correct type. This means that we can construct a PositionF64 with numbers of different type that are convertable to Float64, e.g. `PositionF64(1,1//2)` but we cannot construct `PositionF64` where the fields would be of different type (e.g. `Int`, `Float32`, etc.) or they are not trivially convertable (e.g. `String`).

# Fields in composite types do not have to have a specified type.  We can define a `VaguePosition` without specifying the type
struct VaguePosition
  x 
  y 
end
# This works as the definition above except that the arguments are not converted to `Float64` now. One can store different values in `x` and `y`, for example `String` (e.g. VaguePosition("Hello","world")). Although the above definition might be convenient, it limits the compiler's ability to specialize, as the type  `VaguePosition` does not carry information about type of `x` and `y`, which has a negative impact on the performance. For example
using BenchmarkTools
move(a::T,b::T) where T = T(a.x + b.x, a.y + b.y)
x = [PositionF64(rand(), rand()) for _ in 1:100]
y = [VaguePosition(rand(), rand()) for _ in 1:100]
@benchmark reduce(move, x)
@benchmark reduce(move, y)
# Giving fields of a composite type an abstract type does not really solve the problem of the compiler not knowing the type. In this example, it still does not know, if it should use instructions for `Float64` or `Int8`.
struct LessVaguePosition
  x::Real
  y::Real 
end
z = [LessVaguePosition(rand(), rand()) for _ in 1:100];
@benchmark reduce(move, z)
# While from the perspective of generating optimal code, both definitions are equally informative to the compiler as it cannot assume anything about the code, the  `LessVaguePosition` will ensure that the position will contain only numbers, hence catching trivial errors like instantiating `VaguePosition` with non-numeric types for which arithmetic operators will not be defined (recall the discussion on the  beggining) of the lecture. 

# All structs defined above are immutable (as we have seen above in the case of `Tuple`), which means that one cannot change a field (unless the struct wraps a container, like and array, which allows that). For example this raises an error
a = Position(1, 2)
a.x = 2

# If one needs to make a struct mutable, use the keyword `mutable` befoer keyword `struct` as
mutable struct MutablePosition
  x::Float64
  y::Float64
end
# In mutable structures, we can change the values of fields.
a = MutablePosition(1e0, 2e0)
a.x = 2
# Note, that the memory layout of mutable structures is different, as fields now contain references to memory locations, where the actual values are stored. 

# ### Parametric Types
# So far, we had to trade-off flexibility for generality in type definitions. Can we have both? The answer is affirmative. The way to achieve this  **flexibility** in definitions of the type while being  able to generate optimal code is to  **parametrize** the type definition. This is achieved by replacing types with a parameter (typically a single uppercase character) and decorating in definition by specifying different type in curly brackets. For example 
struct PositionT{T}
  x::T
  y::T 
end
u = [PositionT(rand(), rand()) for _ in 1:100];
@btime reduce(move, u);
# Notice that the compiler can take advantage of specializing for different types (which does not have effect as in modern processrs have addition of `Float` and `Int` takes the same time).
v = [PositionT(rand(1:100), rand(1:100)) for _ in 1:100];
@btime reduce(move, v);
# The above definition suffers the same problem as `VaguePosition`, which is that it allows us to instantiate the `PositionT` with non-numeric types, e.g. `String`. We solve this by restricting the types `T` to be childs of some supertype, in this case `Real`
struct Position{T<:Real}
  x::T
  y::T 
end
# which will throw error if we try to initialize it with `Position("1.0", "2.0")`. 
# 
# Naturally, fields in structures can be of different types, as is in the below pointless example.
struct PositionXY{X<:Real, Y<:Real}
  x::X
  y::Y
end

# ### Abstract parametric types
# Like Composite types, Abstract types can also have parameters. These parameters defines types that are common for all child types. A very good example is Julia's definition of arrays of arbitrary dimension `N` and type `T` of its items as
# ```julia
# abstract type AbstractArray{T,N} end
# ```
# Different `T` and `N` gives rise to different variants of `AbstractArrays`, 
# therefore `AbstractArray{Float32,2}` is different from `AbstractArray{Float64,2}` 
# and from `AbstractArray{Float64,1}.` Note that these are still `Abstract` types, 
# which means you cannot instantiate them. They purpose is
# * to allow to define operations for broad class of concrete types
# * to inform compiler about constant values, which can be used 
# Notice in the above example that parameters of types do not have to be types, but also values of primitive types, as in the above example of `AbstractArray` `N` is the number of dimensions which is an integer value.
# 
# For convenience, it is common to give some important partially instantiated Abstract types an **alias**, for example `AbstractVector` as 
# ```julia
# const AbstractVector{T} = AbstractArray{T,1}
# ```
# is defined in `array.jl:23` (in Julia 1.6.2), which allows us to define for example general prescription for `dot` product of two abstract vectors as 
function dot(a::AbstractVector, b::AbstractVector)
  @assert length(a) == length(b)
  mapreduce(*, +, a, b)
end
# You can verify that the above general function can be compiled to a performant code if 
# specialized for a particular arguments.
@code_native mapreduce(*,+, [1,2,3], [1,2,3])


# ## More on use of types in function definitions
# ### Terminology
# * A *function* refers to a set of "methods" for a different combination of type parameters (a term function can be therefore considered as refering to a mere **name**). *Methods* define different behavior for different type of arguments for a given function. For in below example
move(a::Position, b::Position) = Position(a.x + b.x, a.y + b.y)
move(a::Vector{<:Position}, b::Vector{<:Position}) = move.(a,b)
# `move` refers to a function with methods `move(a::Position, b::Position)` and `move(a::Vector{<:Position}, b::Vector{<:Position})`. When different behavior on different types is defined by a programmer, as shown above, it is also called *implementation specialization*. There is another type of specialization, called compiler specialization*, which occurs when the compiler generates different functions for you from a single method. For example for 
move(Position(1,1), Position(2,2))
move(Position(1.0,1.0), Position(2.0,2.0))
# the compiler generates two methods, one for `Position{Int64}` and the other for `Position{Float64}`. Notice that inside generated functions, the compiler needs to use different intrinsic operations, which can be viewed from 
@code_native  move(Position(1,1), Position(2,2))
# and 
@code_native move(Position(1.0,1.0), Position(2.0,2.0))

# ## Intermezzo: How Julia compiler works?
# Let's walk through an example. Consider following definitions
move(a::Position, by::Position) = Position(a.x + by.x, a.y + by.y)
move(a::T, by::T) where {T<:Position} = Position(a.x + by.x, a.y + by.y)
move(a::Position{Float64}, by::Position{Float64}) = Position(a.x + by.x, a.y + by.y)
move(a::Vector{<:Position}, by::Vector{<:Position}) = move.(a, by)
move(a::Vector{<:Position}, by::Position) = move.(a, by)
# and a function call 
a = Position(1.0, 1.0)
by = Position(2.0, 2.0)
move(a, by)
# 1. The compiler knows that you call function `move` 
# 2. and the compiler infers type of arguments (you can see the result using 
    (typeof(a),typeof(by))
# 3. The compiler identifies all methods that can be applied to a function `move` with arguments of type `(Position{Float64}, Position{Float64})`
    Base.method_instances(move, (typeof(a), typeof(by)))
    m = Base.method_instances(move, (typeof(a), typeof(by))) |> first
# 4a. If the method has been specialized (compiled), then the arguments are prepared and the method is invoked. The compiled specialization can be seen from 
  m.cache
# 4b. If the method has not been specialized (compiled), the compiler compiles the method for a given type of arguments  and continues as in step 4a.
# A compiled function is therefore  a "blob" of **native code** living in a particular memory location. When Julia calls a function, it needs to pick a right block corresponding to a function with particular type of parameters.
#
# If the compiler cannot narrow types of arguments to concrete types, it has to perform the above procedure inside the called function, which has negative effect on the performance, as the type resulution and identification of the method can be slow, especially for methods with many arguments (e.g. 30ns for a method with one argument, 
# 100 ns for method with two arguements).
# Recall the above example 
wolfpack_a =  [Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
@benchmark energy(wolfpack_a)
# and 
wolfpack_b =  Any[Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
@benchmark energy(wolfpack_b)
# An interesting intermediate between fully abstract and fully concrete type happens, when the compiler knows that arguments have abstract type, which is composed of a small number of concrete types. This case  called Union-Splitting, which happens when there is just a little bit of uncertainty. Julia will do something like
# ```julia
# argtypes = typeof(args)
# push!(execution_stack, args)
# if T == Tuple{Int, Bool}
#   @goto compiled_blob_1234
# else # the only other option is Tuple{Float64, Bool}
#   @goto compiled_blob_1236
# end
# ``` 
# For example 
const WolfOrSheep = Union{Wolf, Sheep}
wolfpack_c =  WolfOrSheep[Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
@benchmark energy(wolfpack_c)
# THanks to union splitting, Julia is able to have performant operations on arrays with undefined / missing values for example 
[1, 2, 3, missing] |> typeof

# ### More on matching methods and arguments
# In the above process, the step, where Julia looks for a method instance with corresponding parameters can be very confusing. The rest of this lecture will focus on this. For those who want to have a formal background, we recommend (talk of  Francesco Zappa Nardelli)[https://www.youtube.com/watch?v=Y95fAipREHQ] and / or the that of (Jan Vitek)[https://www.youtube.com/watch?v=LT4AP7CUMAw].
# 
# When Julia needs to specialize a method instance, in needs to find it among multiple definitions. A single function can have many method instances, see for example `methods(+)` which  lists all methods instances of `+` function. How Julia select the proper one?
# 1. It finds all methods where type of arguments match or are subtypes of restrictions  on arguments in the method definition.
# 2a. If there are multiple matches, the compiler selects the most specific definition. 
# 2b. If the compiler cannot decide, which method instance to choose, it throws an error. 
confused_move(a::Position{Float64}, by) = Position(a.x + by.x, a.y + by.y)
confused_move(a, by::Position{Float64}) = Position(a.x + by.x, a.y + by.y)
confused_move(Position(1.0,2.0), Position(1.0,2.0))
# 2c. If it cannot find a suitable method, it throws an error.
move(Position(1,2), VaguePosition("hello","world"))

# Some examples
# Consider following definitions
move(a::Position, by::Position) = Position(a.x + by.x, a.y + by.y)
move(a::T, by::T) where {T<:Position} = T(a.x + by.x, a.y + by.y)
move(a::Position{Float64}, by::Position{Float64}) = Position(a.x + by.x, a.y + by.y)
move(a::Vector{<:Position}, by::Vector{<:Position}) = move.(a, by)
move(a::Vector{T}, by::Vector{T}) where {T<:Position} = move.(a, by)
move(a::Vector{<:Position}, by::Position) = move.(a, by)
# Which method will compiler select for 
move(Position(1.0,2.0), Position(1.0,2.0))
# First three matches types of argumens, but the compiler will select the third one, since it is the most specific.
#
# Which method will compiler select for 
move(Position(1,2), Position(1,2))
# Again, the first and second method definitions match the argument, but the second is the most specific.
#
# Which method will compiler select for 
move([Position(1,2)], [Position(1,2)])
# Again, the fourth and fifth method definitions match the argument, but the fifth is the most specific.
move([Position(1,2), Position(1.0,2.0)], [Position(1,2), Position(1.0,2.0)])

# ### Frequent problems
# 1. Why the following fails?
foo(a::Vector{Real}) = println("Vector{Real}")
foo([1.0,2,3])
#   Julia's type system is **invariant**, which means that `Vector{Real}` is different from `Vector{Float64}` and from `Vector{Float32}`, even though `Float64` and `Float32` are sub-types of `Real`. Therefore `typeof([1.0,2,3])` isa `Vector{Float64}` which is not subtype of `Vector{Real}.` For **covariant** languages, this would be true. For more information on variance in computer languages, see ()[https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)]. If de above definition of `foo` should be applicable to all vectors which has elements of subtype of `Real` we have define it as 
foo(a::Vector{T}) where {T<:Real} = println("Vector{T} where {T<:Real}")
#   or equivalently but more tersely as 
foo(a::Vector{<:Real}) = println("Vector{T} where {T<:Real}")
# 2. Diagonal rule
# rule says that the type repeat in method signature, it has to be a concrete type. Consider for example the function below
move(a::T, b::T) where {T<:Position}
# we cannot call it with `move(Position(1.0,2.0), Position(1,2))`, since in this case `Position(1.0,2.0)` is of type `Position{Float64}` while `Position(1,2)` is of type `Position{Int64}`.
# 3. When debugging why arguments does not match the particular method definition, it is useful to use `typeof`, `isa`, and `<:` commands. For example 
typeof(Position(1.0,2.0))
#
typeof(Position(1,2))
#
Position(1,2) isa Position{Float64}
#
Position(1,2) isa Position{Real}
#
Position(1,2) isa Position{<:Real}
#
typeof(Position(1,2)) <: Position{<:Float64}
#
typeof(Position(1,2)) <: Position{<:Real}


# ### A bizzare definitions which you can encounter
# A following definition of One-Hot Matrix is taken from Flux.jl
struct OneHotArray{T<:Integer, L, N, var"N+1", I<:Union{T,AbstractArray{T, N}}} <: AbstractArray{Bool, var"N+1"}
  indices::I
end
# The parameters of the type carry an information about the type used to encode position of `one` in each column in `T`), the dimension of one-hot vectors in `L`, the dimension of the storage of `indices` in `N` (which is zero for OneHotVector and one for OneHotMatrix), number of dimensions of the OneHotArray in `var"N+1"` and the type of underlying storage of indicies `I`.


