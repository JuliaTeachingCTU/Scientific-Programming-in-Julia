# # Motivation

# Before going into details about Julia type system, we spent few minutes motivating 
# the two main role of type system, which is (i) structuring the code and (ii) and 
# communicating to the compiler your intentions how the type will be used. The 
# first aspect is important for the convenience of programmer and possible abstraction 
# in the language, the latter aspect is important for the speed of generated code. 

# What Wikipedia tells about type and type system?

# * In computer science and computer programming, a **data type** or simply **type** is 
# an attribute of data which tells the compiler or interpreter how the programmer 
# intends to use the data (see ![wiki](https://en.wikipedia.org/wiki/Data_type])).* 

# * A **type system** is a logical system comprising a set of rules that assigns a 
# property called a type to the various constructs of a computer program, such as variables, expressions, functions or modules. These types formalize and enforce the otherwise implicit categories the programmer uses for algebraic data types, data structures, or other components (see ![wiki](https://en.wikipedia.org/wiki/Type_system])).*

# ## Structuring the code
# The main role is therefore aiding help to **structure** the code and impose semantic restriction.
# Consider for example two types with the same definition but different names.

struct Wolf
	name::String
  energy::Int
end

struct Sheep
	name::String
  energy::Int
end

# This allows us to define functions applicable only to the corresponding type
howl(wolf::Wolf) = println(wolf.name, " has howled.")
baa(sheep::Sheep) = println(sheep.name, " has baaed.")

# Therefore the compiler (or interpretter) **enforces** that wolf can only `howl`
# and never `baa` and vice versa sheep can only `baa`. In this sense, it ensures 
# that `howl(sheep)` and `baa(wolf)` never happen.
# For a comparison, consider alternative definition as follows
bark(animal) = println(animal.name, " has howled.")
baa(animal)  = println(animal.name, " has baaed.")
# in which case the burden of ensuring that wolf will not baa rest upon the
# programmer which will inevitably lead to errors.

# ## Intention of use and restrictions on compilers
# The *intention of use* in types is related to how efficient code can compiler 
# produce for that given intention. As an example, consider a following two 
# alternatives to represent a set of animals:
a = [Wolf("1", 1), Wolf("2", 2), Sheep("3", 3)]
b = (Wolf("1", 1), Wolf("2", 2), Sheep("3", 3))
# and define a function to sum energy of all animals as 
energy(animals) = mapreduce(x -> x.energy, +, animals)
# Inspecting the compiled code using 
@code_native energy(a)
@code_native energy(b)
# one observes the second version produces more optimal code. Why is that?
# * In the first representation, `a`, animals are stored in `Array` which can have 
#   arbitrary size and can contain arbitrary animals. This means that compiler has 
#   to compile `energy(a)` such that it works on such arrays.
# * In the second representation, `b`, animals are stored in `Tuple`, 
#   which specializes for lengths and types of items. This means that 
#   the compiler knows the number of animals and the type of each animal 
#   on each position within the tuple, which allows him to specialize.

# This difference will indeed have an impact on the time of code execution. 
# On my i5-8279U CPU, the difference (as measured by BenchmarkTools) is
using BenchmarkTools
@btime energy(a);
@btime energy(b);
# Which nicely demonstrates that smart choice of types can greatly affect the performance.
# Does it mean that we should always use `Tuples` instead of `Arrays`? Surely not, it is 
# just that each is better for different use-case. Using Tuples means that compiler will 
# compile special function for each tuple it observes, which is clearly wasteful.

# # The type system

# ## Julia is dynamicaly typed
# Julia's type system is dynamic, which means that all types are resolved during runtime. 
# **But**, if the compiler can infer types of all variables of a function, it can specialize 
# the function for that given type of variables which lead to an efficient code. Consider a 
# modified example where we represent two wolfpacks:
wolfpack_a =  [Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
wolfpack_b =  Any[Wolf("1", 1), Wolf("2", 2), Wolf("3", 3)]
# `wolfpack_a` carries a type `Vector{Wolf}` while `wolfpack_b` has a type `Vector{Any}`. 
# This means that in the first case, the compiler know that all items are of the type `Wolf`
# and it can specialize functions using this information. In case of `wolfpack_b`, he does 
# not know which animal he will encounter (although all are of the same type), and therefore 
# it needs to dynamically resolve the type of each item upon its use. This ultimately leads 
# to less performant code.
@btime energy(wolfpack_a)
@btime energy(wolfpack_b)

# To conclude, julia is indeed dynamically typed language, **but** if the compiler can infer 
# all types in called function in advance, it does not have to perform a type resolution 
# during execution, which produces a performant code.

# ## Types of types
# Julia divides types into three classes: primitive, composite, and abstract.

# ### Primitive types
# Citing the ![documentation](https://docs.julialang.org/en/v1/manual/types/#Primitive-Types): 
# *A primitive type is a concrete type whose data consists of plain old bits. Classic examples 
# of primitive types are integers and floating-point values. Unlike most languages, Julia
# lets you declare your own primitive types, rather than providing only a fixed set of 
# built-in ones. In fact, the standard primitive types are all defined in the language 
# itself.*

# The definition of primitive types look as follows
primitive type Float16 <: AbstractFloat 16 end
primitive type Float32 <: AbstractFloat 32 end
primitive type Float64 <: AbstractFloat 64 end
# and they are mainly used to jump-start julia's type system. It rarely make a sense to 
# define a special primitive type, as it make sense only if you define special functions 
# operating on its bits, which makes mostly sense if you want to expose special operations 
# provided by underlying CPU / LLVM compiler. For example `+` for `Int32` is different 
# from `+` for `Float32` as they call a different intrinsic operation. You can inspect this 
# jump-starting of type system by yourself by inspecting Julia's source.
# ```julia
# julia> @which +(1,2)
# +(x::T, y::T) where T<:Union{Int128, Int16, Int32, Int64, Int8, UInt128, UInt16, UInt32, UInt64, UInt8} in Base at int.jl:87
# ```

# At `int.jl:87` 
# ```julia
# (+)(x::T, y::T) where {T<:BitInteger} = add_int(x, y)
# ```
# we seen that `+` of integers is calling function `add_int(x, y)`, which is defined in a core 
# part of the compiler in `Intrinsics.cpp` (yes, in C++).

# From Julia docs: *Core is the module that contains all identifiers considered "built in" to 
# the language, i.e. part of the core language and not libraries. Every module implicitly 
# specifies using Core, since you can't do anything without those definitions.*

# Primitive types are rarely used, and they will not be used in this course. We mention them 
# for the sake of completness and refer reader to the official Documentation (and source code 
# of Julia).

# ### [Composite types](@id composite_types)
# The composite types are similar to `struct` in C (they even have the same memory layout). It is not a great idea to think about them as objects (in OOP sense), because objects tie together *data* and *functions* over owned data. Contrary in Julia (as in C), the function operates over data, but are not tied to them. Composite types are a workhorses of Julia's type system, as most user-defined types are composite.

# The composite type is defined as
struct PositionF64
  x::Float64
  y::Float64
end
# which defines a structure with two fields `x` and `y` of type `Float64`. Julia compiler 
# creates a default constructor, where both (but generally all) arguments are converted 
# using `(convert(Float64, x), convert(Float64, y)` to the correct type. This means that 
# we can construct a PositionF64 with numbers of different type that are convertable to 
# Float64, e.g. `PositionF64(1,1//2)` but we cannot construct `PositionF64` where fields 
# would be of different type (e.g. `Int`, `Float32`, etc.).

# Composite types do not have to have a specified type, e.g.
struct VaguePosition
  x 
  y 
end
# which would work as the definition above and allows to store different values in `x`, for 
# example `String`. But it would limit compiler's ability to specialize, which can have a
# negative impact on the performance. For example
using BenchmarkTools
move(a::T, b::T) where {T} = T(a.x + b.x, a.y + b.y)
x = [PositionF64(rand(), rand()) for _ in 1:100]
y = [VaguePosition(rand(), rand()) for _ in 1:100]
@btime reduce(move, x);
@btime reduce(move, y);
# The same holds if the Composite type contains field with AbstractType, for example
struct LessVaguePosition
  x::Real
  y::Real 
end
z = [LessVaguePosition(rand(), rand()) for _ in 1:100];
@btime reduce(move, z);
# While from the perspective of generating optimal code, both definitions are equally 
# informative to the compiler as it cannot assume anything about the code, the 
# `LessVaguePosition` will ensure that the position will contain only numbers, hence
# catching trivial errors like instantiating `VaguePosition` with non-numeric types
# for which arithmetic operators will not be defined (recall the discussion on the 
# beggining) of the lecture. 

# A recommended way to have **flexibility** in definitions of the type while being 
# able to generate optimal code is a **parametrization** of the type definition. This is
# achieved by replacing types with a some variable (typically a single uppercase character) 
# and decorating in definition by specifying different type in curly brackets. For example 
struct PositionT{T}
  x::T
  y::T 
end
u = [PositionT(rand(), rand()) for _ in 1:100];
@btime reduce(move, u);
# Notice that the compiler can take advantage of specializing for differen types 
# (which does not have effect as in modern processrs have addition of Float and Int takes 
# the same time).
v = [PositionT(rand(1:100), rand(1:100)) for _ in 1:100];
@btime reduce(move, v);
# The above definition suffers the same problem as `VaguePosition`, which is that it allows us 
# to instantiate the `PositionT` with non-numeric types, e.g. `String`. We can resolve this 
# by restricting the types `T` can attain as 
struct Position{T<:Real}
  x::T
  y::T 
end
# which will throw error if we try to initialize it with `Position("1.0", "2.0")`.

# All structs defined above are immutable (as we have seen above in the case of `Tuple`), which means that one cannot change a field (unless the struct wraps a container, like and array, which allows that). For example this raises an error
a = Position(1, 2)
a.x = 2

# If one needs to make a struct mutable, use the keyword `mutable` as follows
mutable struct MutablePosition{T}
  x::T
  y::T
end

a = MutablePosition(1e0, 2e0)
a.x = 2
# but there might be performance penalty in some cases (not observable at this simple demo).


# ### Abstract Type 
# 
# The role of abstract type is for structuring the code. They can be viewed as a **set** containing
# types that can be instantiated. This allows to define methods restricted to arguments with 
# **set of types**. This allows to define general methods for set of types where we expect the
# same behavior (recall the Julia design motivation: It it moves like a duck, quaks as a duck, 
# it is a duck). An abstract types are defined by preceding a definition of a type (declared using `struct` 
# keyword) with a keyword `abstract`. For example following set of abstract types defines 
# part of julia's number systems.
# ```julia
# abstract type Number end
# abstract type Real     <: Number end
# abstract type AbstractFloat <: Real end
# abstract type Integer  <: Real end
# abstract type Signed   <: Integer end
# abstract type Unsigned <: Integer end
# ```
# where `<:` means "is a subtype of" and it is used in declarations where the right-hand is an 
# immediate sypertype of a given type (`Integer` has an immediate supertype `Real`.) 
# The abstract type `Number` is derived from `Any` which is a default supertype of any 
# type (this means all subtypes are derived from `Any`).  

# with that, we can define a `sgn` function for **all** real numbers as 
sgn(x::Real) = x > 0 ? 1 : x < 0 ? -1 : 0
# and we know it would be correct for all real numbers. This means that if anyone creates 
# a new subtype of `Real`, the above function can be used. This also means that 
# **it is expected** that comparison operations are defined for any real numbers.

# For unsigned numbers, the `sgn` can be simplified, as it is sufficient to verify if they are different (greated) then zeros, therefore the function can read
sgn(x::Unsigned) = Int(x > 0)
# and again, it applies to all numbers derived from `Unsigned`. Recall that 
# `Unsigned <: Integer <: Real,` how does Julia decides, 
# which version of the function `sgn` to use for `UInt8(0)`? It chooses the most 
# specific version, and therefore for `sgn(UInt8(0))` it will use `sgn(x::Unsinged)`. 
# If the compiler cannot decide, typically it encounters an ambiguity, it throws an error 
# and recommend which function you should define to resolve it.

# The above behavior allows to define default "fallback" implementations and while allowing 
# to specialize for sub-types. A usual example is a matrix multiplication, which has a 
# generic (and slow) implementation with many specializations, which can take advantage 
# of structure (sparse, banded), or of optimized implementations (e.g. blas implementation 
# for dense matrices with eltype `Float32` and `Float64`).

# Again, Julia does not make a difference between abstract types defined in `Base` 
# libraries shipped with the language and those defined by you (the user). All are treated 
# the same.

# (![From Julia documentation](https://docs.julialang.org/en/v1/manual/types/#man-abstract-types))
# Abstract types cannot be instantiated, which means that we cannot create a variable that 
# would have an abstract type (try `typeof(Number(1f0))`). Also, Abstract types cannot have 
# any fields, therefore there is no composition (there are lengty discussions of why this is so,
# one of the most definite arguments of creators is that Abstract types with fields frequently lead
# to children types not using some fields (consider circle vs. ellipse)).

# Like Composite types, Abstract types can have parameters. For example Julia defines an array of arbitrary dimension `N` and type `T` of its items as
# ```julia
# abstract type AbstractArray{T,N} end
# ```
# Different `T` and `N` gives rise to different variants of `AbstractArrays`, 
# therefore `AbstractArray{Float32,2}` is different from `AbstractArray{Float64,2}` 
# and from `AbstractArray{Float64,1}.` Note that these are still `Abstract` types, 
# which means you cannot instantiate them. They purpose is
# * to allow to define operations for broad class of concrete types
# * to inform compiler about constant values, which can be used 

# For convenience, you can name some important partially instantiated Abstract types, 
# for example `AbstractVector` as 
# ```julia
# const AbstractVector{T} = AbstractArray{T,1}
# ```
# is defined in `array.jl:23` (in Julia 1.6.2), which allows us to define for example general 
# prescription for `dot` product of two abstract vectors as 
function dot(a::AbstractVector, b::AbstractVector)
  @assert length(a) == length(b)
  mapreduce(*, +, a, b)
end
# You can verify that the above general function can be compiled to a performant code if 
# specialized for a particular arguments.
@code_native mapreduce(*,+, [1,2,3], [1,2,3])

# ## More on use of types in function definitions
# ### Terminology
# * A *function* refers to a set of "methods" for a different 
# combination of type parameters (a term function can be therefore considered as refering 
# to a mere **name**). A *method* defining different behavior for different type of arguments 
# are also called specializations. For example
move(a, b) = Position(a.x + b.x, a.y + b.y)
move(a::Position, b::Position) = Position(a.x + b.x, a.y + b.y)
move(a::Vector{<:Position}, b::Vector{<:Position}) = move.(a,b)
# `move` refers to function, where `move(a, b)`, 
# `move(a::Position, b::Position)` and `move(a::Vector{<:Position}, b::Vector{<:Position})` 
# are methods. When different behavior on different types is defined by a programmer, 
# as shown above, we call about *implementation 
# specialization*. There is another type of specialization, called *compiler specialization*, 
# which occurs when the compiler generates different functions for you from a single method. For example for 
move(Position(1,1), Position(2,2))
move(Position(1.0,1.0), Position(2.0,2.0))
# the compiler has to generate two methods, since in the first case it will be adding 
# `Int64`s while in the latter it will be adding `Float64`s (and it needs to use different 
# intrinsics, which you can check using `@code_native  move(Position(1,1), Position(2,2))` 
# and `@code_native move(Position(1.0,1.0), Position(2.0,2.0))`).

### How Julia compiler works?
# Compiled function is a "blob" of **native code** living in a particular memory location. 
# When Julia calls a function, it needs to pick a right block corresponding to a function 
# with particular type of parameters.
# Calling a function involves therefore involves
# * preparing parameters
# * finding a right block corresponding to a function with particular type of parameters.
# The process can be made during runtime (when code is executing) or during compile time 
# (when code is compiled) and everything in between.

# An interesting intermediate is called Union-Splitting, which happens when there is just 
# a little bit of uncertainty. Julia will do something like
# ```julia
# argtypes = typeof(args)
# push!(execution_stack, args)
# if T == Tuple{Int, Bool}
#   @goto compiled_blob_1234
# else # the only other option is Tuple{Float64, Bool}
#   @goto compiled_blob_1236
# end
# ``` 

# 1. The compiler knows that you want to call function `move` 
# 2. The compiler tries to infer type of arguments, the result can be viewed using `typeof(a)` / `typef(b)`
# 3. The compiler looks of he has already specialized (compiled) a version of `move` for give 
#     types of parameters.
# 4a. If the method has been specialized, it is called adter arguments are prepared.
# 4b. If the method has not been specialized, the compiler find a method instance corresponding 
#     to type of parameters, compile it (and cache with all method instances called witihn this 
#     method), and execute it.
# If the compiler determines that the arguments are abstract, it has to perform the above procedure
# within the function, which has negative effect on the performance, as the above procedure can 
# be slow, especially for methods with many arguments (e.g. 30ns for a method with one argument, 
# 100 ns for method with two arguements).
# 
# In the above process, the step, where Julia looks for a method instance with corresponding 
# parameters can be very confusing. The rest of this lecture will focus on this. For those who 
# want to know more, we recommend 
# !(talk of  Francesco Zappa Nardelli)[https://www.youtube.com/watch?v=Y95fAipREHQ]
# and / or the talk of !(Jan Vitek)[https://www.youtube.com/watch?v=LT4AP7CUMAw].
# 
# When Julia needs to specialize a method instance, in needs to find it among multiple definitions.
# A single function can have many method instances, see for example `methods(+)` which 
# lists all methods instances of `+` function. How Julia select the proper one?
# 1. It finds all methods where type of arguments match or are subtypes of restrictions 
#    on arguments in the method definition.
# 2. If there are multiple matches, the compiler selects the most specific definition. 
# 3. If the compiler cannot decide, which method instance to choose, it throws an error. 

# Example:
# Consider following definitions
move(a::Position, by::Position) = Position(a.x + by.x, a.y + by.y)
move(a::T, by::T) where {T<:Position} = Position(a.x + by.x, a.y + by.y)
move(a::Position{Float64}, by::Position{Float64}) = Position(a.x + by.x, a.y + by.y)
move(a::Vector{<:Position}, by::Vector{<:Position}) = move.(a, by)
move(a::Vector{<:Position}, by::Position) = move.(a, by)
# and function call 
a = Position(1.0, 1.0)
by = Position(2.0, 2.0)
move(a, by)
# since types of `a` and `b` are `Position{Float64}`, from the above list following three
# methods 
# ```julia
# move(a::Position, by::Position) = Position(a.x + by.x, a.y + by.y)
# move(a::T, by::T) where {T<:Position} = Position(a.x + by.x, a.y + by.y)
# move(a::Position{Float64}, by::Position{Float64}) = Position(a.x + by.x, a.y + by.y)
# ```
# can be used for the given arguments. Julia will select ** the most specific ** version, which is 
# ```julia
# move(a::Position{Float64}, by::Position{Float64}) = Position(a.x + by.x, a.y + by.y)
# ```
# For 
a = Position(1f0, 1f0)
by = Position(2f0, 2f0)
move(a, by)
# following methods matche types of arguments
# ```julia
# move(a::Position, by::Position) = Position(a.x + by.x, a.y + by.y)
# move(a::T, by::T) where {T<:Position} = Position(a.x + by.x, a.y + by.y)
# ```
# from which Julia picks 
# ```julia
# move(a::T, by::T) where {T<:Position} = Position(a.x + by.x, a.y + by.y)
# ```
# which is again more specific.

# ### Frequent problems
# 1. Why the following fails?
foo(a::Vector{Real}) = println("Vector{Real}")
foo([1.0,2,3])
# Julia's type system is **invariant**, which means that `Vector{Real}` is different from 
# `Vector{Float64}` and from `Vector{Float32}`, even though `Float64` and `Float32` are 
# sub-types of `Real`. Therefore `typeof([1.0,2,3])` isa `Vector{Float64}` which is not 
# subtype of `Vector{Real}.` For **covariant** languages, this would be true. For more 
# information on variance in computer languages, see 
# !()[https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)].
# If de above definition of `foo` should be applicable to all vectors which has elements
# of subtype of `Real` we have define it as 
foo(a::Vector{T}) where {T<:Real} = println("Vector{T} where {T<:Real}")
# or equivalently but more tersely as 
foo(a::Vector{<:Real}) = println("Vector{T} where {T<:Real}")

# 2. Diagonal rule
# rule says that the type repeat in method signature, it has to be 
# a concrete type. Consider for example the function below
move(a::T, b::T) where {T<:Position}
# we cannot call it with `move(Position(1.0,2.0), Position(1,2))`,
# since in this case `Position(1.0,2.0)` is of type `Position{Float64}`
# while `Position(1,2)` is of type `Position{Int64}`.
