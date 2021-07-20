### The power of Type System \& multiple dispatch
  - Zero cost abstraction the compiler has information about types and can freely inline
    `x |> identity |> identity |> identity |> identity |> identity`
  - Why the type system is important for efficiency
  - Bad practices 
  - **LABS:**
      + Number type-tree
      + Design Interval Aritmetics (for ODES)


### Mutable struct

Exaplain a difference between mutable struct and non-mutable from the point of view of boxing and indiference

Expain not-properly typed struct

### Abstract types
* why `Vector{AbstractFloat}` is a bad idea, while `Vector{Float64}` is a good one?
* Explain Boxing
* why `Vector{AbstractFloat}` is different to `Vector{<:AbstractFloat}` or `Vector{T} where {T<:AbstractFloat}` 


### Why [1,2,3] is not a Vector{Number}
Just to extract some relevant points because the manual is quite dense:

Vector{Number} is a concrete type, and [1, 2, 3] has type Vector{Int}, which is also a concrete type. One concrete type is never a subtype of another concrete type, they are the leaves of the type tree.

Vector{Number} is concrete, even though Number is not a concrete type. That’s because it has a concrete implementation which can store all types that are subtypes of Number, it has a specific memory layout etc. On the other hand, AbstractVector{Int} is the other way around and not a concrete type, because the container is abstract even though the element is concrete.

What you can do instead is [1, 2, 3] isa Vector{<:Number} which is true. That’s because <:Number is a sort of placeholder which means “any type which is a subtype of Number”. This is often needed for dispatching on containers where you want to allow set of element types. f(x::Vector{Number}) can take only arguments of type Vector{Number}, whereas g(x::Vector{<:Number}) can take e.g. Vector{Int}, Vector{Float64}, Vector{Real}, Vector{Number}, etc.


https://youtu.be/Y95fAipREHQ?t=958c
### Alignment in the memory
* Super-importnat for speed, cache utilization
* Know if you are row or column major 
* Vector{Vector{Float}} is a really bad idea over 

### A Headache examples
* 
```
function Base.reduce(::typeof(hcat), xs::Vector{TV})  where {T, L, TV<:OneHotLike{T, L}}
  OneHotMatrix(reduce(vcat, map(_indices, xs)), L)
end
```

### Alocation 
* show reduction of `Vector{Vector{Int}}`



### Turning multiple-distpatch to single dispatch

https://stackoverflow.com/questions/39133424/how-to-create-a-single-dispatch-object-oriented-class-in-julia-that-behaves-l/39150509#39150509


### conversion among variables

reinterpret(Float32, ref_sketch)

### Trivia 
why this is false Vector{Int} <: AbstractVector{<:Any}
