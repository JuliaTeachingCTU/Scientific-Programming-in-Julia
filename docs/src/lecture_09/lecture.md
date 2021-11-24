# Generated functions

## Motivation 
Sometimes, especially as we will see later, it is handy to generate function once the types with which the function is called are known. For example if we have function `foo(args...)`, we can generate different body for different length of `Tuple` and types in `args`. Wait, do we need generated functions for this? Not really, as 
- we can deal with variability of `args` using normal control-flow logic `if length(args) == 1 elseif ...`
- we can (automatically) generate (a possibly infinite) set of functions `foo` specialized for each length of args (or combination of types of args) and let multiple dispatch to deal with this
- we cannot deal with this situation with macros, because macros do not see types, only parsed AST, which is in this case always the same.

Generated functions elegantly deals with that situation, as we can specialize the code for a given type of argumnets. Generated functions, like macros, therefore **return expressions** and not **results**. But unlike macros, they do not have access to values of arguments, but to their types (the arguments are of type `Type`). They are also called when compiler needs (which means at least once for each combination of arguments, but possibly more times due to code invalidation).

Let's look at example
```julia
@generated function genplus(x, y)
  println("generating genplus(x, y)")
  @show (x, y, typeof(x), typeof(y))
  quote 
    println("executing generated genplus(x, y)")
    @show (x, y, typeof(x), typeof(y))
    x + y
  end
end
```
now observe the output
```julia
generating genplus(x, y)
(x, y, typeof(x), typeof(y)) = (Float64, Float64, DataType, DataType)
executing generated genplus(x, y)
(x, y, typeof(x), typeof(y)) = (1.0, 1.0, Float64, Float64)
true

julia> genplus(1.0, 1.0) == 1.0 + 1.0
executing generated genplus(x, y)
(x, y, typeof(x), typeof(y)) = (1.0, 1.0, Float64, Float64)
true

julia> genplus(1, 1) == 1 + 1
generating genplus(x, y)
(x, y, typeof(x), typeof(y)) = (Int64, Int64, DataType, DataType)
executing generated genplus(x, y)
(x, y, typeof(x), typeof(y)) = (1, 1, Int64, Int64)
true
```
which shows that the body of `genplus` is called for each combination of types of parameters, but the generated code is called whenever `genplu` is called.

Generated functions has to be pure in the sense that they are not allowed to have side effects, for example modifying some global variables. Note that printing is not allowed in pure functions, as it modifies the global buffer. This rule is not strict, including things, but not obeying it can lead to unexpected errors, as you do not know, at which moment the functions will be called.

Finally, generated functions cannot call functions that has been defined after their definition.
```julia
@generated function genplus(x, y)
  foo()
  :(x + y)
end

foo() = println("foo")
genplus(1,1)
```
throws an error saying *The applicable method may be too new: running in world age 29640, while current world is 29641*, where the *applicable method* is the `foo`.

### An example that explains everything
Consider a version of `map` applicable on `NamedTuple`s with permuted names. Recall the behavior of normal map, which works if names are the same
```julia
x = (a = 1, b = 2, c = 3)
y = (a = 4, b = 5, c = 6)
map(+, x, y)
```
but it does not work with permuted names
```julia
x = (a = 1, b = 2, c = 3)
y = (c = 6, b = 5, a = 4)
map(+, x, y)
```
How to fix this? The usual approach would be to iterate over the keys in named tuples as 
```julia
function permuted_map(f, x::NamedTuple{KX}, y::NamedTuple{KY}) where {KX, KY}
    @assert issubset(KX,KY)
    NamedTuple{KX}(map(k -> f(x[k], y[k]), KX))
end
```
But, can we do better? Recall that in `NamedTuple`s, we exactly know the position of the arguments, hence we should be able to directly match the corresponding arguments without using `get`. 

Since creation (and debugging) of generated functions is difficult, we start with a single single-argument unrolled map
```julia
@generated function unrolled_map(f, x::NamedTuple{KX}) where {KX} 
    vals = [:(f(getfield(x, $(QuoteNode(k))))) for k in KX]
    :(($(vals...),))
end
```
We see inserting `Symbol` specifying the field in the NamedTuple is a bit tricky. It needs to be quoted, since `$()` which is needed to substitute `k` for its value "peels" one layer of the quoting. Compare it to `vals = [:(f(getfield(x, $(k)))) for k in KX]`

Since getting the field is awkward, we write syntactic sugar for that
```julia
_get(name, k) = :(getfield($(name), $(QuoteNode(k))))
```
with that, we proceed to a nicer two argument function which we have desired
```julia
@generated function unrolled_map(f, x::NamedTuple{KX}, y::NamedTuple{KY}) where {KX, KY} 
    @assert issubset(KX,KY)
    _get(name, k) = :(getfield($(name), $(QuoteNode(k))))
    vals = [:(f($(_get(:x, k)), $(_get(:y, k)))) for k in KX]
    :(NamedTuple{$(KX)}(($(vals...),)))
end
```
We can check that the `unrolled_map` unrolls the map and generates just needed operations
```julia
julia> @code_typed unrolled_map(+, x, y)
CodeInfo(
1 ─ %1  = Main.getfield(x, :a)::Int64
│   %2  = Main.getfield(y, :a)::Int64
│   %3  = Base.add_int(%1, %2)::Int64
│   %4  = Main.getfield(x, :b)::Int64
│   %5  = Main.getfield(y, :b)::Int64
│   %6  = Base.add_int(%4, %5)::Int64
│   %7  = Main.getfield(x, :c)::Int64
│   %8  = Main.getfield(y, :c)::Int64
│   %9  = Base.add_int(%7, %8)::Int64
│   %10 = %new(NamedTuple{(:a, :b, :c), Tuple{Int64, Int64, Int64}}, %3, %6, %9)::NamedTuple{(:a, :b, :c), Tuple{Int64, Int64, Int64}}
└──       return %10
) => NamedTuple{(:a, :b, :c), Tuple{Int64, Int64, Int64}}
```
and compare this to the same of 
```julia
@code_typed permuted_map(+, x, y)
```
which is not shown here for the sake of conciseness.

For fun, we can create a version which replaces the `Symbol` arguments directly by position numbers
```julia
@generated function unrolled_map(f, x::NamedTuple{KX}, y::NamedTuple{KY}) where {KX, KY} 
    @assert issubset(KX,KY)
    _get(name, k, KS) = :(getfield($(name), $(findfirst(k .== KS))))
    vals = [:(f($(_get(:x, k, KX)), $(_get(:y, k, KY)))) for k in KX]
    :(NamedTuple{$(KX)}(($(vals...),)))
end
```
## Example with getindex from docs


## Optionally generated functions
Let's now observe, how the macro `@generated` is expanded. 
```julia
julia> @macroexpand @generated function gentest(x)
           return :(x + x)
       end

:(function gentest(x)
      if $(Expr(:generated))
          return $(Expr(:copyast, :($(QuoteNode(:(x + x))))))
      else
          $(Expr(:meta, :generated_only))
          return
      end
  end)
```
It is expanded into the function with else, where first branch `$(Expr(:generated))` generates the expression `:(x + x)` and copy it the return and the other spits the error saying that the function is has only generated version. This suggest the possibility (and reality) to implement two versions of the same function, one generated and one normal and leave it up to the compiler to decide which to take. It is entirely up to the author to ensure that both versions are the same. Which version the compiler will take? The last comment on [23168](https://github.com/JuliaLang/julia/pull/23168) (as of time of writing) states

*Currently the @generated branch is always used. In the future, which branch is used will mostly depend on whether the JIT compiler is enabled and available, and if it's not available, then it will depend on how much we were able to compile before the compiler was taken away. So I think it will mostly be a concern for those that might need static compilation and JIT-less deployment.* )

## Contextual dispatch / overdubbing
Imagine that under some circumstances (context), you would like to use alternative implementations of some  functions. One of the most cited motivations for this is automatic differentiation, where you would like to take the code **as-is** and calculate gradients with respect to some variables. Other motivations mentioned by `Cassette.jl` implementing this technique *includes dynamic code analysis (e.g. profiling, rr-style debugging, etc.), JIT compilation to new hardware/software backends, automatic differentiation, interval constraint programming, automatic parallelization/rescheduling, automatic memoization, lightweight multistage programming, execution graph extraction, and more.*
In theory, we can do all the above by directly modifying the code or introducing new types, but that may require a lot of coding and changing of foreign libraries.

The technique we desire is called contextual dispatch, which means that under some context, we invoke a different function. The library `Casette.jl` provides a high-level api for overdubbing, but it is by no means interesting to see, how it works, as it shows, how we can "interact" with the lowered code before the code is typed.

## insertion of the code

Imagine that julia has compiled some function. For example 
```julia
foo(x,y) = x * y + sin(x)
```
and observe its lowered SSA format
```julia
julia> @code_lowered foo(1.0, 1.0)
CodeInfo(
1 ─ %1 = x * y
│   %2 = Main.sin(x)
│   %3 = %1 + %2
└──      return %3
)
```
The lowered form is very nice, because on the left hand, there is **always** one variable and right-hand side is simplified to have (mostly) single call / expression. Moreover, in the lowered form, all control flow operations like `if`, `for`, `while` and exceptions are converted to `Goto` and `GotoIfNot`, which simplifies their handling. 

### Codeinfo
We can access lowered form by
```julia
ci = @code_lowered foo(1.0, 1.0)
```
which returns an object of type `CodeInfo` containing many fields [docs](https://docs.julialang.org/en/v1/devdocs/ast/#Lowered-form). To make the investigation slightly more interesting, we modify the function a bit to have local variables as 
```julia
julia> function foo(x,y) 
  z = x * y 
  z + sin(x)
end

julia> ci = @code_lowered foo(1.0, 1.0)
CodeInfo(
1 ─      z = x * y
│   %2 = z
│   %3 = Main.sin(x)
│   %4 = %2 + %3
└──      return %4
)
```
The most important (and interesting) is `code`
```julia
julia> ci.code
4-element Vector{Any}:
 :(_4 = _2 * _3)
 :(_4)
 :(Main.sin(_2))
 :(%2 + %3)
 :(return %4)
```
containing expressions corresponding to each line of the lowered form. You are free to access them (and modify them with care). Variables identified underscore `Int`, for example `_2`, are slotted variables which are variables which has name in the code, that come through arguments or through an explicit assignment `:(=)`. Namsed of slotted variables are stored in `ci.slotnames` and they are of type 
```julia
julia> typeof(ci.code[1].args[2].args[2])
Core.SlotNumber

julia> ci.slotnames[ci.code[1].args[2].args[2].id]
:x

julia> ci.slotnames[ci.code[1].args[2].args[3].id]
:y

julia> ci.slotnames[ci.code[1].args[1].id]
:z
```
Remaining variables are identified by and integer with prefix `%`, where the number corresponds to the line (index in `ci.code`), where the variable was created. For example the fourth line `:(%2 + %3)` adds the results of the second line `:(_4)` containing variable `z` and the third line `:(Main.sin(_2))`. Each slot variable as type stored in `slottypes`, which provides some information about how the variable is used ([see docs](https://docs.julialang.org/en/v1/devdocs/ast/#CodeInfo)). Note that if you modify / introduce slot variables, the length of `slotnames` and `slottypes` has to match and it has to be equal to the maximum number of slotted variable.

`CodeInfo` also contains information about the source code. Each item of `ci.code` has an identifier in `ci.codelocs` which is an index into `ci.linetable` containing `Core.LineInfoNode` identifying lines in source code (or inserted through REPL). Notice that `ci.linetable` is generally shorter then `ci.codelocs`, as one line of source code can be translated to multiple lines in lowered code. 

The important feature of lowered form is that we can freely edit (create new) `CodeInfo` and that generated function can return `CodeInfo` object instead of the AST but you need to **explicitly** write the `return` statement ([see issue 25678](https://github.com/JuliaLang/julia/issues/25678)).

### Strategy for overdubbing
In overdubbing, our intention is to recursively dive into called function definitions and modify / change their code. In our working example below, on which we will demonstrate the manual implementation (for educational purposes), our goal would be to enclose each function call by statements that would log the exection time. This means we would like to implement a simplified recording Profiler. This functionality cannot be implemented by a macros, since macro will not allow is to dive into the function. For example in our example of function `foo`, we would would not be able to dive into the definition of `sin` function (not that this is terribly good idea, but the point should be clear).

The overdubbing pattern works as follows.
1. We define a `@generated function overdub(f, args...)` which takes as a first argument function `f` and then its arguments.
2. In the function `overdub` we retrieve the `CodeInfo` for `f(args...)`, which is possible as we know at this moments type of arguments.
3. We modify the the `CodeInfo` of `f(args...)` according to our liking. Importantly, we replace all function calls `some_fun(some_args...)` with `overdub(some_fun, some_args...)` which establishes the recursive pattern
4. Modify the arguments of the `CodeInfo` of `f(args...)` to match `overdub(f, args..)`
5. Return the modified codeinfo.


#### The profiler
The implementation of the simplified logging profiler is straightforward and looks as follows.
```julia
module LoggingProfiler

struct Calls
    stamps::Vector{Float64} # contains the time stamps
    event::Vector{Symbol}  # name of the function that is being recorded
    startstop::Vector{Symbol} # if the time stamp corresponds to start or to stop
    i::Ref{Int}
end

function Calls(n::Int)
    Calls(Vector{Float64}(undef, n+1), Vector{Symbol}(undef, n+1), Vector{Symbol}(undef, n+1), Ref{Int}(0))
end

global const to = Calls(100)

function Base.show(io::IO, calls::Calls)
    offset = 0
    for i in 1:calls.i[]
        offset -= calls.startstop[i] == :stop
        foreach(_ -> print(io, " "), 1:max(offset, 0))
        rel_time = calls.stamps[i] - calls.stamps[1]
        println(io, calls.event[i], ": ", rel_time)
        offset += calls.startstop[i] == :start
    end
end


"""
    record_start(ev::Symbol)

    record the start of the event, the time stamp is recorded after all counters are 
    appropriately increased
"""
function record_start(ev::Symbol)
    calls = Main.to
    n = calls.i[] = calls.i[] + 1
    n > length(calls.stamps) && return 
    calls.event[n] = ev
    calls.startstop[n] = :start
    calls.stamps[n] = time_ns()
end

"""
    record_end(ev::Symbol)

    record the end of the event, the time stamp is recorded before all counters are 
    appropriately increased
"""
function record_end(ev::Symbol)
    t = time_ns()
    calls = Main.to
    n = calls.i[] = calls.i[] + 1
    n > length(calls.stamps) && return 
    calls.event[n] = ev
    calls.startstop[n] = :stop
    calls.stamps[n] = t
end

reset!(calls::Calls) = calls.i[] = 0

function Base.resize!(calls::Calls, n::Integer)
  resize!(calls.stamps, n)
  resize!(calls.event, n)
  resize!(calls.startstop, n)
end
end
```

The important functions are `report_start` and `report_end` which marks the beggining and end of the executed function. They differ mainly when time is recorded (on the end or on the start of the function call). The profiler has fixed capacity to prevent garbage collection, which might be increased.


Let's now describe the individual parts of `overdub` function and then we will present it in its entirety.
At first, we retrieve the codeinfo `ci` of overdubbed function. For now, we will just assume we obtain it for example by
```julia
ci = @code_lowered foo(1.0, 1.0)
```
we initialize the new `CodeInfo` object by emptying some dummy function as 
```julia
dummy() = return
new_ci = code_lowered(dummy, Tuple{})[1]
empty!(new_ci.code)
empty!(new_ci.slotnames)
empty!(new_ci.linetable)
empty!(new_ci.codelocs)
new_ci
```

Then, we need to copy the slot variables from the `ci` codeinfo of `foo` functions to the new codeinfo but we also needs to add arguments of function `overdub(f, args...)` since the compiler sees `overdub(f, args...)` and not `foo(x,y)`. We do this by 
```julia
new_ci.slotnames = vcat([Symbol("#self#"), :f, :args], ci.slotnames[2:end])
new_ci.slotflags = vcat([0x00, 0x00, 0x00], ci.slotflags[2:end])
```
where we also filled the `slotflags`. Authors admit that names `:f` and `:args` in the above should be replaced by `gensym`ed name, but they do not anticipate this code to be used for some bigger problems where name-clashes might occur.
We also copy information about lines from the source code as 
```julia
foreach(s -> push!(new_ci.linetable, s), ci.linetable)
```
The most difficult part when rewriting the `CodeInfo` objects is working with indexes, as the line numbers and left hand side variables are strictly ordered one by one and we need to properly change the indexes to reflect changes we made. We will therefore keep three lists
```julia
maps = (
    ssa = Dict{Int, Int}(),
    slots = Dict{Int, Any}(),
    goto = Dict{Int,Int}(),
    )
```
where 
- `slots` maps slot variables in `ci` to that in `new_ci`
- `ssa` maps indexes of left-hand side assignments in `ci` to that in `new_ci`
- `goto` maps lines to which `GotoNode` and `GotoIfNot` points to variables in `ci` to that in `new_ci` (in our profiler example, we need to ensure to jump on the beggining of logging of executions)
mapping of slots can be initialized in advance, as it is a static shift by `2` 
```julia
maps.slots[1] = Core.SlotNumber(1)
foreach(i -> maps.slots[i] = Core.SlotNumber(i + 2), 2:length(ci.slotnames)) 
```
and we can check the correctness by
```julia
@assert all(ci.slotnames[i] == new_ci.slotnames[maps.slots[i].id] for i in 1:length(ci.slotnames))  #test that 
```
Equipped with that, we start rewriting the code of `foo(x, y)`. We start by a small preample, where we assign values of `args...` to `x`, and `y.` For the same of further simplicity, we map the slotnames to either `Core.SlotNumber` or to `Core.SSAValues` which simplifies the rewriting logic a bit.
```julia
newci_no = 0
for i in 1:length(args)
    newci_no +=1
    push!(new_ci.code, Expr(:call, Base.getindex, Core.SlotNumber(3), i))
    maps.slots[i+1] = Core.SSAValue(newci_no)
    push!(new_ci.codelocs, ci.codelocs[1])
end
```
Now we come to the pinnacle of rewriting the body of the `foo(x,y)` as 
```julia
for (ci_no, ex) in enumerate(ci.code)
    if timable(ex)
        fname = exportname(ex)
        push!(new_ci.code, Expr(:call, GlobalRef(LoggingProfiler, :record_start), fname))
        push!(new_ci.codelocs, ci.codelocs[ci_no])
        newci_no += 1
        maps.goto[ci_no] = newci_no
        ex = overdubbable(ex) ? Expr(:call, GlobalRef(Main, :overdub), ex.args...) : ex
        push!(new_ci.code, ex)
        push!(new_ci.codelocs, ci.codelocs[ci_no])
        newci_no += 1
        maps.ssa[ci_no] = newci_no
        push!(new_ci.code, Expr(:call, GlobalRef(LoggingProfiler, :record_end), fname))
        push!(new_ci.codelocs, ci.codelocs[ci_no])
        newci_no += 1
    else
        push!(new_ci.code, ex)
        push!(new_ci.codelocs, ci.codelocs[ci_no])
        newci_no += 1
        maps.ssa[ci_no] = newci_no
    end
end
```
where the important parts are 
- depending on the type of expressions (controlled by `timable`) we decide, if function's execution time should be recorded;
- `fname = exportname(ex)` obtains the name of the profiled function call
- `push!(new_ci.code, Expr(:call, GlobalRef(LoggingProfiler, :record_start), fname))` records the start of the exection
- `maps.goto[ci_ssa_no] = ssa_no` updates the map from the code line number in `ci` to that in rewritten `new_ci`
- `maps.ssa[ci_ssa_no] = ssa_no` updates the map from the ssa line number in `ci` to that in rewritten `new_ci`
- `ex = overdubbable(ex) ? Expr(:call, GlobalRef(Main, :overdub), ex.args...) : ex` modifies the function call (expression in general) to recurse the overdubbing
Finally, we need to change the names of slot variables (`Core.SlotNumber`) and variables indexed by ssa (`Core.SSAValue`).
```julia
for i in length(args)+1:length(new_ci.code)
    new_ci.code[i] = remap(new_ci.code[i], maps)
end
```
where `remap` is a following block of code
```julia
remap(ex::Expr, maps) = Expr(ex.head, remap(ex.args, maps)...)
remap(args::AbstractArray, maps) = map(a -> remap(a, maps), args)
remap(c::Core.GotoNode, maps) = Core.GotoNode(maps.goto[c.label])
remap(c::Core.GotoIfNot, maps) = Core.GotoIfNot(remap(c.cond, maps), maps.goto[c.dest])
remap(r::Core.ReturnNode, maps) = Core.ReturnNode(remap(r.val, maps))
remap(a::Core.SlotNumber, maps) = maps.slots[a.id]
remap(a::Core.SSAValue, maps) = Core.SSAValue(maps.ssa[a.id])
remap(a::Core.NewvarNode, maps) = Core.NewvarNode(maps.slots[a.slot.id])
remap(a::GlobalRef, maps) = a
remap(a::QuoteNode, maps) = a
remap(ex, maps) = ex
```

!!! warn 
    ### Retrieving the code is a bit tricky. Consider a following function:
    ```julia
    function test(x::T) where T<:Union{Float64, Float32}
       x < T(pi)
    end

    julia> ci = @code_lowered test(1.0)
    CodeInfo(
    1 ─ %1 = ($(Expr(:static_parameter, 1)))(Main.pi)
    │   %2 = x < %1
    └──      return %2
    )
    ```
    the `Expr(:static_parameter, 1)` in the first line of code obtains the type parameter `T` of the function `test`. Since this information is not accessible in the `CodeInfo`, it might render our tooling useless. The need hook is `Base.Meta.partially_inline!` which partially inlines this into the `CodeInfo` object.
    The code to retrieve the `CodeInfo` adapted from `IRTools` is little convolved as 

    ```julia
    function retrieve_code_info(sigtypes, world = Base.get_world_counter())
        S = Tuple{map(s -> Core.Compiler.has_free_typevars(s) ? typeof(s.parameters[1]) : s, sigtypes)...}
        _methods = Base._methods_by_ftype(S, -1, world)
        if isempty(_methods) 
            @info("method $(sigtypes) does not exist")
            return(nothing)
        end
        type_signature, raw_static_params, method = _methods[1]
        mi = Core.Compiler.specialize_method(method, type_signature, raw_static_params, false)
        ci = Base.isgenerated(mi) ? Core.Compiler.get_staged(mi) : Base.uncompressed_ast(method)
        Base.Meta.partially_inline!(ci.code, [], method.sig, Any[raw_static_params...], 0, 0, :propagate)
        ci
    end
    ```
    but
    ```julia
    julia> ci = retrieve_code_info((typeof(test), Float64))
    CodeInfo(
        @ REPL[5]:2 within `test'
    1 ─ %1 = ($(QuoteNode(Float64)))(Main.pi)
    │   %2 = x < %1
    └──      return %2
    )
    ```
    it performs the needed inlining of Float64
A further complication is that we need to change variables back and also gives them names. if names have existed before. Recall that lowered form introduces additional variables while converting the code to SSA. The variables defined in the source code (argument names and user-defined variables) can be found in `ci.slotnames`,
whereas the variables introduces during lowering to SSA are named by the line number.
Observe a difference between
```julia
julia> foo(x,y) = x * y + sin(x)
foo (generic function with 1 method)

julia> retrieve_code_info((typeof(foo), Float64, Float64)).slotnames
3-element Vector{Symbol}:
 Symbol("#self#")
 :x
 :y
```
and
```julia
julia> function foo(x, y)
       z = x * y
       z + sin(y)
       end
foo (generic function with 1 method)

julia> retrieve_code_info((typeof(foo), Float64, Float64)).slotnames
4-element Vector{Symbol}:
 Symbol("#self#")
 :x
 :y
 :z
```


```
macro meta(ex)
  isexpr(ex, :call) || error("@meta f(args...)")
  f, args = ex.args[1], ex.args[2:end]
  :(meta(typesof($(esc.((f, args...))...))))
end
```


```julia 
macro timeit(ex::Expr)
    ex.head != :call && error("timeit is implemented only for function calls") 
    quote
        ctx = Context()
        
    end
end
macro timeit(ex)
    error("timeit is implemented only for function calls") 
end
```


## Petite zygote
```julia
world = Base.get_world_counter()
sigtypes = (typeof(+), Int, Int)
sigtypes = (typeof(sin), Int)

function retrieve_code_info(sigtypes, world = Base.get_world_counter())
  S = Tuple{map(s -> Core.Compiler.has_free_typevars(s) ? typeof(s.parameters[1]) : s, sigtypes)...}
  _methods = Base._methods_by_ftype(S, -1, world)
  isempty(_methods) && @error("method $(sigtypes) does not exist, may-be run it once")
  type_signature, raw_static_params, method = _methods[1] # method is the same as we would get by invoking methods(+, (Int, Int)).ms[1]  

  # this provides us with withe CodeInfo
  method_instance = Core.Compiler.specialize_method(method, type_signature, raw_static_params, false)
  code_info = Core.Compiler.retrieve_code_info(method_instance)
end
```
[details of code info](https://docs.julialang.org/en/v1/devdocs/ast/#CodeInfo)


```julia
foo(x) = 5x + 3
foo(x,y) = x * y + sin(x)
foo(1.0,1.0)
sigtypes = (typeof(foo), Float64, Float64)
sigtypes = (typeof(foo), Float64, Float64)
```

```julia
lowered_code = retrieve_code_info((typeof(foo), Float64))
lowered_code = retrieve_code_info((typeof(foo), Float64, Float64))
lowered_code.code        # contains expression of individual lines of code
lowered_code.slotnames   # contains names of corresponding variables
```

Let's list for which expressions we have rrules in ChainRules
```julia
using ChainRules, ChainRulesCore
rename_args(ex::Expr, primal_names, lowered_fun) = Expr(ex.head, rename_args(ex.args, primal_names, lowered_fun))

function rename_args(args::Vector, primal_names, lowered_fun)
   map(args) do a 
    rename_args(a, primal_names, lowered_fun)
  end
end

function rename_args(a, primal_names, lowered_fun)
  if a isa Core.SlotNumber
    return(lowered_fun.slotnames[a.id])
  elseif a isa Core.SSAValue
    return(primal_names[a.id])
  else
    return(a)
  end
end

rename_args(a::Core.ReturnNode, primal_names, lowered_fun) = Core.ReturnNode(rename_args(a.val, primal_names, lowered_fun))

function generate_grad(lowered_code)
  primal_names = Dict{Int,Symbol}()
  primals = []
  pullbacks = []
  pullback_names = Dict{Int,Any}()
  rename_args(ex) = rename_args(ex, primal_names, lowered_code)
  output_id = -1

  for(exno, ex) in enumerate(lowered_code.code)
    if ex isa Core.ReturnNode
      output_id = ex.val.id
      continue
    end

    if !hasproperty(ex, :head) || ex.head != :call
      push!(primals, rename_args(ex))
      continue
    end

    ex.head != :call && return(rename_args(ex))
    fun_T = :(typeof($(ex.args[1]))) |> eval
    args_T = map(i -> Number, ex.args[2:end])
    _methods = Base._methods_by_ftype(Tuple{typeof(rrule), fun_T, args_T...}, -1, world)
    isempty(_methods) && return(rename_args(ex)) # If the method does not exist, I need to dive_in
    renamed_args = rename_args(ex.args)

    primal   = Symbol("r", exno)
    pullback = Symbol("∂", exno)
    rr = Symbol("rr",exno)
    primal_names[exno] = primal
    pullback_names[exno] = (pullback, renamed_args...)

    # let's work out the pullback
    push!(primals, :($(rr) = rrule($(renamed_args...))))
    push!(primals, :($(primal) = $(rr)[1]))
    push!(primals, :($(pullback) = $(rr)[2]))
  end

  #Let's get the name of the output variable and iniate the pullback
  primal_out = Symbol(:r, output_id) 
  pullback_arg = Symbol(:∂r, output_id) 
  created_grads = Indices{Symbol}()
  for line_no in sort(collect(keys(pullback_names)), rev = true)
    pullfun  = pullback_names[line_no][1]
    pullout = Symbol(:Δ, line_no)
    p = Symbol(:∂r, line_no) 
    push!(pullbacks, :($(pullout) = $(pullfun)($(p))))
    for (i, x) in (enumerate(pullback_names[line_no][3:end]))
      o = Symbol(:∂, x) 
      if o ∈ created_grads
        push!(pullbacks, :($(o) += $(pullout)[$(i+1)]))
      else
        push!(pullbacks, :($(o) = $(pullout)[$(i+1)]))
        insert!(created_grads, o)
      end
    end
  end

  #add gradients with respect to arguments
  argnames = lowered_code.slotnames[2:end]
  ∂args = map(s -> Symbol(:∂, s), argnames)

  quote
    function ∂foo($(argnames...))
      $(primals...)
      function pullback($(pullback_arg))
        $(pullbacks...)
        return(NoTangent(), $(∂args...))
      end
      return($(primal_out), pullback)
    end
  end |> Base.remove_linenums!
end

p, back = ∂foo(1, 1)
p ≈ foo(1,1)
back(1)
```

### some thoughts
ri is the variable
∂i is a function to which I need to pass the ∂ri to get the gradinets 

# TODO

* show constant gradient of linear function in LLVM
