# Manipulating Intermediate Represenation (IR)

```@setup lec09
using InteractiveUtils: @code_typed, @code_lowered, code_lowered
```

## Generated functions
Sometimes it is convenient to generate function once types of arguments are known. For example if we have function `foo(args...)`, we can generate different body for different length of `Tuple` and types in `args`. Do we really need such thing, or it is just wish of curious programmer? Not really, as 
- we can deal with variability of `args` using normal control-flow logic `if length(args) == 1 elseif ...`
- we can (automatically) generate (a possibly infinite) set of functions `foo` specialized for each length of `args` (or combination of types of `args`) and let multiple dispatch to deal with this
- we cannot deal with this situation with macros, because macros do not see types, only parsed AST, which is in this case always the same.

Generated functions allow to specialize the code for a given type of argumnets. They are like macros in the sense that they **return expressions** and not **results**. But unlike macros, the input is not expression or value of arguments, but their types (the arguments are of type `Type`). They are also called when compiler needs (which means at least once for each combination of arguments, but possibly more times due to code invalidation).

Let's look at an example
```@example lec09
@generated function genplus(x, y)
  println("generating genplus(x, y)")
  @show (x, y, typeof(x), typeof(y))
  quote 
    println("executing generated genplus(x, y)")
    @show (x, y, typeof(x), typeof(y))
    x + y
  end
end
nothing # hide
```
and observe the output
```julia
julia> genplus(1.0, 1.0) == 1.0 + 1.0
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
which shows that the body of `genplus` is called for each combination of types of parameters, but the generated code is called whenever `genplus` is called. We can observe how the number of specializations changes with `first(methods(genplus)).specializations`

Generated functions has to be pure in the sense that they are not allowed to have side effects, for example modifying some global variables. Note that printing is not allowed in pure functions, as it modifies the global buffer. From the above example this rule does not seems to be enforced, but not obeying it can lead to unexpected errors mostly caused by not knowing when and how many times the functions will be called.

Finally, generated functions cannot call functions that has been defined after their definition.
```@repl lec09
@generated function genplus(x, y)
  foo()
  :(x + y)
end

foo() = println("foo")
genplus(1,1)
```
Here, the *applicable method* is `foo`.

### An example that explains everything.
Consider a version of `map` applicable to `NamedTuple`s with permuted names.
Recall the behavior of normal map, which works if the names are in the same order.
```@repl lec09
x = (a = 1, b = 2, c = 3)
y = (a = 4, b = 5, c = 6)
map(+, x, y)
```
The same does not work with permuted names:
```@repl lec09
x = (a = 1, b = 2, c = 3)
y = (c = 6, b = 5, a = 4)
map(+, x, y)
```
How to fix this? The usual approach would be to iterate over the keys in named tuples:
```@example lec09
function permuted_map(f, x::NamedTuple{KX}, y::NamedTuple{KY}) where {KX, KY}
    @assert issubset(KX,KY)
    NamedTuple{KX}(map(k -> f(x[k], y[k]), KX))
end
nothing # hide
```
But, can we do better? Recall that in `NamedTuple`s, we exactly know the position of the arguments, hence we should be able to directly match the corresponding arguments without using `get`. 

Since creation (and debugging) of generated functions is difficult, we start with a single-argument unrolled map.
```@repl
@generated function unrolled_map(f, x::NamedTuple{KX}) where {KX} 
    vals = [:(f(getfield(x, $(QuoteNode(k))))) for k in KX]
    :(($(vals...),))
end
unrolled_map(e->e+1, x)
```
We see that inserting a `Symbol` specifying the field in the `NamedTuple` is a
bit tricky. It needs to be quoted, since `$()` which is needed to substitute
`k` for its value "peels" one layer of the quoting. Compare this to
```@repl
KX = (:a, :b, :c)
vals = [:(f(getfield(x, $(k)))) for k in KX]
```

Since getting the field is awkward, we write syntactic sugar for that
```julia
_get(name, k) = :(getfield($(name), $(QuoteNode(k))))
```
with that, we proceed to a nicer two argument function which we have desired:
```@repl lec09
@generated function unrolled_map(f, x::NamedTuple{KX}, y::NamedTuple{KY}) where {KX, KY} 
    @assert issubset(KX,KY)
    _get(name, k) = :(getfield($(name), $(QuoteNode(k))))
    vals = [:(f($(_get(:x, k)), $(_get(:y, k)))) for k in KX]
    :(NamedTuple{$(KX)}(($(vals...),)))
end
nothing # hide
```
We can check that the `unrolled_map` unrolls the map and generates just needed operations
```@repl lec09
@code_typed debuginfo=:none unrolled_map(+, x, y)
```
and compare this to the code generated by the non-generated version `permuted_map`:
```julia
@code_typed debuginfo=:none permuted_map(+, x, y)
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

## Optionally generated functions
Macro `@generated` is expanded to
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
which is a function with an if-condition, where the first branch `$(Expr(:generated))` generates the expression `:(x + x)` and returns it. The other spits out an error saying that the function has only a generated version. This suggests the possibility (and reality) that one can implement two versions of the same function; A generated and a *normal* version. It is left up to the compiler to decide which one to use. It is entirely up to the author to ensure that both versions are the same. Which version will the compiler take? The last comment on [23168](https://github.com/JuliaLang/julia/pull/23168) (as of time of writing) states:

"*Currently the `@generated` branch is always used. In the future, which branch is used will mostly depend on whether the JIT compiler is enabled and available, and if it's not available, then it will depend on how much we were able to compile before the compiler was taken away. So I think it will mostly be a concern for those that might need static compilation and JIT-less deployment.*"

## Contextual dispatch / overdubbing
Imagine that under some circumstances (context), you would like to use alternative implementations of some functions. One of the most cited motivations for this is automatic differentiation, where you would like to take the code **as-is** and calculate gradients with respect to some variables. Other use cases of this approach are mentioned in `Cassette.jl`:

"*Downstream applications for Cassette include dynamic code analysis (e.g. profiling, rr-style debugging, etc.), JIT compilation to new hardware/software backends, automatic differentiation, interval constraint programming, automatic parallelization/rescheduling, automatic memoization, lightweight multistage programming, graph extraction, and more.*"

In theory, we can do all the above by directly modifying the code or introducing new types, but that may require a lot of coding and changing of foreign libraries.

The technique we desire is called contextual dispatch, which means that under some context, we invoke a different function. The library `Casette.jl` / `IRTools` provides a high-level API for overdubbing, but it is interesting to see, how it works, as it shows, how we can "interact" with the lowered code before the code is typed.

### Insertion of code

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
The lowered form is convenient, because on the left hand, there is **always** one variable and the right-hand side is simplified to have (mostly) a single call / expression. Moreover, in the lowered form, all control flow operations like `if`, `for`, `while` and exceptions are converted to `Goto` and `GotoIfNot`, which simplifies their handling. 

### Codeinfo
We can access the lowered form by
```julia
ci = @code_lowered foo(1.0, 1.0)
```
which returns an object of type `CodeInfo` containing many fields [docs](https://docs.julialang.org/en/v1/devdocs/ast/#Lowered-form). To make the investigation slightly more interesting, we modify the function a bit to have local variables:
```@repl lec09
function foo(x,y) 
  z = x * y 
  z + sin(x)
end

ci = @code_lowered foo(1.0, 1.0)
```
The most important (and interesting) field is `code`:
```@repl lec09
ci.code
```
It contains expressions corresponding to each line of the lowered form. You are free to access them (and modify them with care). Variables identified with underscore `Int`, for example `_2`, are slotted variables which are variables which have a name in the code, defined via input arguments or through an explicit assignment `:(=)`. The names of slotted variables are stored in `ci.slotnames` and they are of type 
```@repl lec09
typeof(ci.code[1].args[2].args[2])
ci.slotnames[ci.code[1].args[2].args[2].id]
ci.slotnames[ci.code[1].args[2].args[3].id]
ci.slotnames[ci.code[1].args[1].id]
```
The remaining variables are identified by an integer with prefix `%`, where the number corresponds to the line (index in `ci.code`), in which the variable was created. For example the fourth line `:(%2 + %3)` adds the results of the second line `:(_4)` containing variable `z` and the third line `:(Main.sin(_2))`. The type of each slot variable is stored in `slottypes`, which provides some information about how the variable is used ([see docs](https://docs.julialang.org/en/v1/devdocs/ast/#CodeInfo)). Note that if you modify / introduce slot variables, the length of `slotnames` and `slottypes` has to match and it has to be equal to the maximum number of slotted variables.

`CodeInfo` also contains information about the source code. Each item of `ci.code` has an identifier in `ci.codelocs` which is an index into `ci.linetable` containing `Core.LineInfoNode` identifying lines in the source code (or in the REPL). Notice that `ci.linetable` is generally shorter then `ci.codelocs`, as one line of source code can be translated to multiple lines in lowered code. 

The important feature of the lowered form is that we can freely edit (create new) `CodeInfo` and that generated functions can return a `CodeInfo` object instead of the AST. However, you need to **explicitly** write a `return` statement ([see issue 25678](https://github.com/JuliaLang/julia/issues/25678)).

### Strategy for overdubbing
In overdubbing, our intention is to recursively dive into called function definitions and modify / change their code. In our example below, with which we will demonstrate the manual implementation (for educational purposes), our goal is to enclose each function call with statements that log the execution time. This means we would like to implement a simplified recording profiler. This functionality cannot be implemented by a macros, since macros do not allow us to dive into function definitions. For example, in our function `foo`, we would would not be able to dive into the definition of `sin` (not that this is a terribly good idea, but the point should be clear).

The overdubbing pattern works as follows.
1. We define a `@generated function overdub(f, args...)` which takes as a first argument a function `f` and then its arguments.
2. In the function `overdub` we retrieve the `CodeInfo` for `f(args...)`, which is possible as we know types of the arguments at this time.
3. We modify the the `CodeInfo` of `f(args...)` according to our preference. Importantly, we replace all function calls `some_fun(some_args...)` with `overdub(some_fun, some_args...)` which establishes the recursive pattern.
4. Modify the arguments of the `CodeInfo` of `f(args...)` to match `overdub(f, args..)`.
5. Return the modified `CodeInfo`.

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

function Base.show(io::IO, calls::Calls)
    offset = 0
    if calls.i[] >= length(calls.stamps)
        @warn "The recording buffer was too small, consider increasing it"
    end
    for i in 1:min(calls.i[], length(calls.stamps))
        offset -= calls.startstop[i] == :stop
        foreach(_ -> print(io, " "), 1:max(offset, 0))
        rel_time = calls.stamps[i] - calls.stamps[1]
        println(io, calls.event[i], ": ", rel_time)
        offset += calls.startstop[i] == :start
    end
end

global const to = Calls(100)

"""
    record_start(ev::Symbol)

    record the start of the event, the time stamp is recorded after all counters are 
    appropriately increased
"""
record_start(ev::Symbol) = record_start(to, ev)
function record_start(calls, ev::Symbol)
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
record_end(ev::Symbol) = record_end(to, ev::Symbol)
function record_end(calls, ev::Symbol)
    t = time_ns()
    n = calls.i[] = calls.i[] + 1
    n > length(calls.stamps) && return 
    calls.event[n] = ev
    calls.startstop[n] = :stop
    calls.stamps[n] = t
end

reset!() = to.i[] = 0

function Base.resize!(calls::Calls, n::Integer)
  resize!(calls.stamps, n)
  resize!(calls.event, n)
  resize!(calls.startstop, n)
end
end
```

The important functions are `report_start` and `report_end` which mark the beggining and end of the executed function. They differ mainly when time is recorded (on the end or on the start of the function call). The profiler has a fixed capacity to prevent garbage collection, which might be increased.

Let's now describe the individual parts of `overdub` before presenting it in its entirety.
At first, we retrieve the codeinfo `ci` of the overdubbed function. For now, we will just assume we obtain it for example by
```julia
ci = @code_lowered foo(1.0, 1.0)
```
we initialize the new `CodeInfo` object by emptying some dummy function as 
```@example lec09
dummy() = return
new_ci = code_lowered(dummy, Tuple{})[1]
empty!(new_ci.code)
empty!(new_ci.slotnames)
empty!(new_ci.linetable)
empty!(new_ci.codelocs)
new_ci
```

Then, we need to copy the slot variables from the `ci` codeinfo of `foo` to the new codeinfo. Additionally, we have to add the arguments of `overdub(f, args...)` since the compiler sees `overdub(f, args...)` and not `foo(x,y)`:
```@repl lec09
new_ci.slotnames = vcat([Symbol("#self#"), :f, :args], ci.slotnames[2:end])
new_ci.slotflags = vcat([0x00, 0x00, 0x00], ci.slotflags[2:end])
```
Above, we also filled the `slotflags`. Authors admit that names `:f` and `:args` in the above should be replaced by a `gensym`ed name, but they do not anticipate this code to be used outside of this educative example where name-clashes might occur.
We also copy information about the lines from the source code:
```@repl lec09
foreach(s -> push!(new_ci.linetable, s), ci.linetable)
```
The most difficult part when rewriting `CodeInfo` objects is working with indexes, as the line numbers and left hand side variables are strictly ordered one by one and we need to properly change the indexes to reflect changes we made. We will therefore keep three lists
```@example lec09
maps = (
    ssa = Dict{Int, Int}(),
    slots = Dict{Int, Any}(),
    goto = Dict{Int,Int}(),
)
nothing # hide
```
where 
- `slots` maps slot variables in `ci` to those in `new_ci`
- `ssa` maps indexes of left-hand side assignments in `ci` to `new_ci`
- `goto` maps lines to which `GotoNode` and `GotoIfNot` point to variables in `ci` to `new_ci` (in our profiler example, we need to ensure to jump on the beggining of logging of executions)
Mapping of slots can be initialized in advance, as it is a static shift by `2` :
```julia
maps.slots[1] = Core.SlotNumber(1)
foreach(i -> maps.slots[i] = Core.SlotNumber(i + 2), 2:length(ci.slotnames)) 
```
and we can check the correctness by
```julia
@assert all(ci.slotnames[i] == new_ci.slotnames[maps.slots[i].id] for i in 1:length(ci.slotnames))  #test that 
```
Equipped with that, we start rewriting the code of `foo(x, y)`. We start by a small preample, where we assign values of `args...` to `x`, and `y`. For the sake of simplicity, we map the slotnames to either `Core.SlotNumber` or to `Core.SSAValues` which simplifies the rewriting logic a bit.
```julia
newci_no = 0
for i in 1:length(args)
    newci_no +=1
    push!(new_ci.code, Expr(:call, Base.getindex, Core.SlotNumber(3), i))
    maps.slots[i+1] = Core.SSAValue(newci_no)
    push!(new_ci.codelocs, ci.codelocs[1])
end
```
Now we come to the pinnacle of rewriting the body of `foo(x,y)` while inserting calls to the profiler:
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
which yields
```julia
julia> new_ci.code
15-element Vector{Any}:
 :((getindex)(_3, 1))
 :((getindex)(_3, 2))
 :(_4 = _2 * _3)
 :(_4)
 :(Main.LoggingProfiler.record_start(:sin))
 :(Main.overdub(Main.sin, _2))
 :(Main.LoggingProfiler.record_end(:sin))
 :(Main.LoggingProfiler.record_start(:+))
 :(Main.overdub(Main.:+, %2, %3))
 :(Main.LoggingProfiler.record_end(:+))
 :(return %4)
```
The important parts are:
- Depending on the type of expressions (controlled by `timable`) we decide, if a function's execution time should be recorded.
- `fname = exportname(ex)` obtains the name of the profiled function call.
- `push!(new_ci.code, Expr(:call, GlobalRef(LoggingProfiler, :record_start), fname))` records the start of the exection.
- `maps.goto[ci_ssa_no] = ssa_no` updates the map from the code line number in `ci` to the one in `new_ci`.
- `maps.ssa[ci_ssa_no] = ssa_no` updates the map from the SSA line number in `ci` to `new_ci`.
- `ex = overdubbable(ex) ? Expr(:call, GlobalRef(Main, :overdub), ex.args...) : ex` modifies the function call (expression in general) to recurse the overdubbing.
Finally, we need to change the names of slot variables (`Core.SlotNumber`) and variables indexed by the SSA (`Core.SSAValue`).
```julia
for i in length(args)+1:length(new_ci.code)
    new_ci.code[i] = remap(new_ci.code[i], maps)
end
```
where `remap` is defined by the following block of code
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
    ### Retrieving the code properly

    Consider the following function:
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
    the `Expr(:static_parameter, 1)` in the first line of code obtains the type parameter `T` of the function `test`. Since this information is not accessible in the `CodeInfo`, it might render our tooling useless. The needed hook is `Base.Meta.partially_inline!` which partially inlines this into the `CodeInfo` object.
    The code to retrieve the `CodeInfo` adapted from `IRTools` is a little involved:

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
    it performs the needed inlining of `Float64`.

## Implementing the profiler with IRTools
The above implementation of the profiler has shown, that rewriting IR manually is doable, but requires a lot of careful book-keeping. `IRTools.jl` makes our life much simpler, as they take away all the needed book-keeping and let us focus on what is important.

```@repl lec09
using IRTools
function foo(x, y)
   z =  x * y
   z + sin(y)
end;
ir = @code_ir foo(1.0, 1.0)
```
We can see that at first sight, the representation of the lowered code in IRTools is similar to that of `CodeInfo`. Some notable differences:
- `SlotNumber` are converted to `SSAValues`
- SSA form is divided into blocks by `GotoNode` and `GotoIfNot` in the parsed `CodeInfo`
- SSAValues do not need to be ordered. The reordering is deffered to the moment when one converts `IRTools.Inner.IR` back to the `CodeInfo`.

Let's now use the IRTools to insert the timing statements into the code for `foo`:
```julia
using IRTools: xcall, insert!, insertafter!

ir = @code_ir foo(1.0, 1.0)
for (v, ex) in ir
    if timable(ex.expr)
        fname = exportname(ex.expr)
        insert!(ir, v, xcall(LoggingProfiler, :record_start, fname))
        insertafter!(ir, v, xcall(LoggingProfiler, :record_end, fname))
    end
end

julia> ir
1: (%1, %2, %3)
  %7 = Main.LoggingProfiler.record_start(:*)
  %4 = %2 * %3
  %8 = Main.LoggingProfiler.record_end(:*)
  %9 = Main.LoggingProfiler.record_start(:sin)
  %5 = Main.sin(%3)
  %10 = Main.LoggingProfiler.record_end(:sin)
  %11 = Main.LoggingProfiler.record_start(:+)
  %6 = %4 + %5
  %12 = Main.LoggingProfiler.record_end(:+)
  return %6
```

Observe that the statements are on the right places but they are not ordered.
We can turn the `ir` object into an anonymous function
```julia
f = IRTools.func(ir)
LoggingProfiler.reset!()
f(nothing, 1.0, 1.0)
LoggingProfiler.to
```
where we can observe that our profiler is working as it should. But this is not yet our final goal. Originally, our goal was to recursivelly dive into the nested functions. IRTools offers a macro `@dynamo`, which is similar to `@generated` but simplifies our job by allowing to return the `IRTools.Inner.IR` object and it also taking care of properly renaming the arguments. With that we write
```julia
using IRTools: @dynamo
profile_fun(f::Core.IntrinsicFunction, args...) = f(args...)
profile_fun(f::Core.Builtin, args...) = f(args...)

@dynamo function profile_fun(f, args...)
    ir = IRTools.Inner.IR(f, args...)
    for (v, ex) in ir
        if timable(ex.expr)
            fname = exportname(ex.expr)
            insert!(ir, v, xcall(LoggingProfiler, :record_start, fname))
            insertafter!(ir, v, xcall(LoggingProfiler, :record_end, fname))
        end
    end
    for (x, st) in ir
        recursable(st.expr) || continue
        ir[x] = xcall(profile_fun, st.expr.args...)
    end
    return ir
end
```
where the first pass is as it was above and the `ir[x] = xcall(profile_fun, st.expr.args...)` ensures that the profiler will recursively call itself. `recursable` is a filter defined as below, which is used to prevent profiling itself (and possibly other things).
```julia
recursable(gr::GlobalRef) = gr.name ∉ [:profile_fun, :record_start, :record_end]
recursable(ex::Expr) = ex.head == :call && recursable(ex.args[1])
recursable(ex) = false
```
Additionally, the first two definitions of `profile_fun` for `Core.IntrinsicFunction` and for `Core.Builtin` prevent trying to dive into functions which do not have a Julia IR. And that's all. The full code is 
```example lec09
using IRTools
using IRTools: var, xcall, insert!, insertafter!, func, recurse!, @dynamo
include("loggingprofiler.jl")
LoggingProfiler.resize!(LoggingProfiler.to, 10000)

function timable(ex::Expr) 
    ex.head != :call && return(false)
    length(ex.args) < 2 && return(false)
    ex.args[1] isa Core.GlobalRef && return(true)
    ex.args[1] isa Symbol && return(true)
    return(false)
end
timable(ex) = false

recursable(gr::GlobalRef) = gr.name ∉ [:profile_fun, :record_start, :record_end]
recursable(ex::Expr) = ex.head == :call && recursable(ex.args[1])
recursable(ex) = false

exportname(ex::GlobalRef) = QuoteNode(ex.name)
exportname(ex::Symbol) = QuoteNode(ex)
exportname(ex::Expr) = exportname(ex.args[1])
exportname(i::Int) = QuoteNode(Symbol("Int(",i,")"))

profile_fun(f::Core.IntrinsicFunction, args...) = f(args...)
profile_fun(f::Core.Builtin, args...) = f(args...)

@dynamo function profile_fun(f, args...)
    ir = IRTools.Inner.IR(f, args...)
    for (v, ex) in ir
        if timable(ex.expr)
            fname = exportname(ex.expr)
            insert!(ir, v, xcall(LoggingProfiler, :record_start, fname))
            insertafter!(ir, v, xcall(LoggingProfiler, :record_end, fname))
        end
    end
    for (x, st) in ir
        recursable(st.expr) || continue
        ir[x] = xcall(profile_fun, st.expr.args...)
    end
    # recurse!(ir)
    return ir
end

macro record(ex)
    esc(Expr(:call, :profile_fun, ex.args...))
end

LoggingProfiler.reset!()
@record foo(1.0, 1.0)
LoggingProfiler.to
```
where you should notice the long time the first execution of `@record foo(1.0, 1.0)` takes. This is caused by the compiler specializing for every function into which we dive into. The second execution of `@record foo(1.0, 1.0)` is fast. It is also interesting to observe how the time of the compilation is logged by the profiler. The output of the profiler `to` is not shown here due to the length of the output.

## Petite Zygote
`IRTools.jl` were created for `Zygote.jl` --- Julia's source-to-source AD system currently powering `Flux.jl`. An interesting aspect of `Zygote` was to recognize that TensorFlow is in its nutshell a compiler, PyTorch is an interpreter. So the idea was to let Julia's compiler compile the gradient and perform optimizations that are normally performed with normal code. Recall that a lot of research went into how to generate efficient code and it is reasonable to use this research. `Zygote.jl` provides mainly reversediff, but there was an experimental support for forwarddiff.

One of the questions when developing an AD engine is where and how to create a computation graph. Recall that in TensorFlow, you specify it through a domain specific language, in PyTorch it generated on the fly. Mike Innes' idea was use SSA form provided by the julia compiler. 
```julia
julia> @code_lowered foo(1.0, 1.0)
CodeInfo(
1 ─      z = x * y
│   %2 = z
│   %3 = Main.sin(y)
│   %4 = %2 + %3
└──      return %4
)
```
It is very easy to differentiate each line, as they correspond to single expressions (or function calls) and importantly, each variable is assigned exactly once. The strategy to use it for AD would as follows.

### Strategy
We assume to have a set of AD rules (e.g. ChainRules), which for a given function returns its evaluation and pullback. If `Zygote.jl` is tasked with computing the gradient.
1. If a rule exists for this function, directly return the rule.
2. If not, deconstruct the function into a sequence of functions using `CodeInfo` / IR representation
3. Replace statements by calls to obtain the evaluation of the statements and the pullback.
4. Chain pullbacks in reverse order.
5. Return the function evaluation and the chained pullback.

### Simplified implementation
The following code is adapted from [this example](https://github.com/FluxML/IRTools.jl/blob/master/examples/reverse.jl)

```julia
using IRTools, ChainRules
using IRTools: @dynamo, IR, Pipe, finish, substitute, return!, block, blocks,
  returnvalue, arguments, isexpr, xcall, self, stmt

struct Pullback{S,T}
  data::T
end

Pullback{S}(data) where S = Pullback{S,typeof(data)}(data)

function primal(ir, T = Any)
  pr = Pipe(ir)
  calls = []
  ret = []
  for (v, st) in pr
    ex = st.expr
    if isexpr(ex, :call)
      t = insert!(pr, v, stmt(xcall(Main, :forward, ex.args...), line = st.line))
      pr[v] = xcall(:getindex, t, 1)
      J = push!(pr, xcall(:getindex, t, 2))
      push!(calls, v)
      push!(ret, J)
    end
  end
  pb = Expr(:call, Pullback{T}, xcall(:tuple, ret...))
  return!(pr, xcall(:tuple, returnvalue(block(ir, 1)), pb))
  return finish(pr), calls
end

@dynamo function forward(m...)
  ir = IR(m...)
  ir == nothing && return :(error("Non-differentiable function ", repr(args[1])))
  length(blocks(ir)) == 1 || error("control flow is not supported")
  return primal(ir, Tuple{m...})[1]
end

```
where 
- the generated function `forward` calls `primal` to perform AD manual chainrule
- actual chainrule is performed in the for loop
- every function call is replaced  `xcall(Main, :forward, ex.args...)`, which is the recursion we have observed above. `stmt` allows to insert information about lines in the source code).
- the output of the forward is the value of the function, and *pullback*, the function calculating gradient with respect to its inputs.
- `pr[v] = xcall(:getindex, t, 1)` fixes the output of the overwritten function call to be the output of `forward(...)`
- the next line logs the *pullback* 
- `Expr(:call, Pullback{T}, xcall(:tuple, ret...))` will serve to call generated function which will assemble the pullback in the right order

Let's now observe how the the IR of `foo` is transformed
```julia
ir = IR(typeof(foo), Float64, Float64)
julia> primal(ir)[1]
1: (%1, %2, %3)
  %4 = Main.forward(Main.:*, %2, %3)
  %5 = Base.getindex(%4, 1)
  %6 = Base.getindex(%4, 2)
  %7 = Main.forward(Main.sin, %3)
  %8 = Base.getindex(%7, 1)
  %9 = Base.getindex(%7, 2)
  %10 = Main.forward(Main.:+, %5, %8)
  %11 = Base.getindex(%10, 1)
  %12 = Base.getindex(%10, 2)
  %13 = Base.tuple(%6, %9, %12)
  %14 = (Pullback{Any, T} where T)(%13)
  %15 = Base.tuple(%11, %14)
  return %15
```
- Every function call was transformed into the sequence of `forward(...)` and obtaining first and second item from the returned typle.
- Line `%14` constructs the `Pullback`, which (as will be seen shortly below) will allow to generate the pullback for the generated function
- Line `%15` generates the returned tuple, where the first item is the function value (computed at line `%11`) and pullback (constructed at libe `%15`).

We define few AD rules by specializing `forward`  with calls from `ChainRules`
```julia
forward(::typeof(sin), x)    = ChainRules.rrule(sin, x)
forward(::typeof(*), x, y)   = ChainRules.rrule(*, x, y)
forward(::typeof(+), x, y)   = ChainRules.rrule(+, x, y)
```
Zygote implements this inside the generated function, such that whatever is added to `ChainRules` is automatically reflected. The process is not as trivial (see [`has_chain_rule`](https://github.com/FluxML/Zygote.jl/blob/master/src/compiler/chainrules.jl)) and for the brevity is not shown here. 

We now obtain the value and the pullback of function `foo` as 
```julia
julia> v, pb = forward(foo, 1.0, 1.0);

julia> pb(1.0)
(0, 1.0, 1.5403023058681398)
```
- The pullback contains in `data` field with individual jacobians that have been collected in `ret` in `primal` function.
```julia
pb.data[1]
pb.data[2]
pb.data[3]
```
The function for which the Jacobian has been created is stored in type parameter `S` of the `Pullback` type. The pullback for `foo` is generated in another generated function, as `Pullback` `struct` is a functor. This is an interesting **design pattern**, which allows us to return *closure* from a generated function. 

Let's now investigate the code generating code for pullback.
```julia

_sum() = 0
_sum(x) = x
_sum(x...) = xcall(:+, x...)

function pullback(pr)
  ir = empty(pr)
  grads = Dict()
  grad(x) = _sum(get(grads, x, [])...)
  grad(x, x̄) = push!(get!(grads, x, []), x̄)
  grad(returnvalue(block(pr, 1)), IRTools.argument!(ir))
  data = push!(ir, xcall(:getfield, self, QuoteNode(:data)))
  _, pbs = primal(pr)
  pbs = Dict(pbs[i] => push!(ir, xcall(:getindex, data, i)) for i = 1:length(pbs))
  for v in reverse(keys(pr))
    ex = pr[v].expr
    isexpr(ex, :call) || continue
    Δs = push!(ir, Expr(:call, pbs[v], grad(v)))
    for (i, x) in enumerate(ex.args)
      grad(x, push!(ir, xcall(:getindex, Δs, i)))
    end
  end
  return!(ir, xcall(:tuple, [grad(x) for x in arguments(pr)]...))
end

@dynamo function (pb::Pullback{S})(Δ) where S
  return pullback(IR(S.parameters...))
end
```
Let's walk how the reverse is constructed for `pr = IR(typeof(foo), Float64, Float64)`
```julia
ir = empty(pr)
grads = Dict()
grad(x) = _sum(get(grads, x, [])...)
grad(x, x̄) = push!(get!(grads, x, []), x̄)
```
construct the empty `ir` for the constructed pullback, defines `Dict` where individual contributors of the gradient with respect to certain variable will be stored, and two function for pushing statements to to `grads`. The next statement
```julia
grad(returnvalue(block(pr, 1)), IRTools.argument!(ir))
```
pushes to `grads` statement that the gradient of the output of the primal `pr` is provided as an argument of the pullback `IRTools.argument!(ir)`. 
```
data = push!(ir, xcall(:getfield, self, QuoteNode(:data)))
_, pbs = primal(pr)
pbs = Dict(pbs[i] => push!(ir, xcall(:getindex, data, i)) for i = 1:length(pbs))
```
sets `data` to the `data` field of the `Pullback` structure containing pullback functions. Then it create a dictionary `pbs`, where the output of each call in the primal (identified by the line) is mapped to the corresponding pullback, which is now a line in the IR representation.
The IR so far looks as 
```julia
1: (%1)
  %2 = Base.getfield(IRTools.Inner.Self(), :data)
  %3 = Base.getindex(%2, 1)
  %4 = Base.getindex(%2, 2)
  %5 = Base.getindex(%2, 3)
```
and `pbs` contains 
```julia
julia> pbs
Dict{IRTools.Inner.Variable, IRTools.Inner.Variable} with 3 entries:
  %6 => %5
  %4 => %3
  %5 => %4
```
says that the pullback of a function producing variable at line `%6` in the primal is stored at variable `%5` in the contructed pullback.
The real deal comes in the for loop 
```julia
for v in reverse(keys(pr))
  ex = pr[v].expr
  isexpr(ex, :call) || continue
  Δs = push!(ir, Expr(:call, pbs[v], grad(v)))
  for (i, x) in enumerate(ex.args)
    grad(x, push!(ir, xcall(:getindex, Δs, i)))
  end
end
```
which iterates the primal `pr` in the reverse order and for every call, it inserts statement to calls the appropriate pullback `Δs = push!(ir, Expr(:call, pbs[v], grad(v)))` and adds gradients with respect to the inputs to values accumulating corresponding gradient in the loop `for (i, x) in enumerate(ex.args) ...`
The last line
```julia
return!(ir, xcall(:tuple, [grad(x) for x in arguments(pr)]...))
```
puts statements accumulating gradients with respect to individual variables to the ir.

The final generated IR code looks as
```julia
julia> pullback(IR(typeof(foo), Float64, Float64))
1: (%1)
  %2 = Base.getfield(IRTools.Inner.Self(), :data)
  %3 = Base.getindex(%2, 1)
  %4 = Base.getindex(%2, 2)
  %5 = Base.getindex(%2, 3)
  %6 = (%5)(%1)
  %7 = Base.getindex(%6, 1)
  %8 = Base.getindex(%6, 2)
  %9 = Base.getindex(%6, 3)
  %10 = (%4)(%9)
  %11 = Base.getindex(%10, 1)
  %12 = Base.getindex(%10, 2)
  %13 = (%3)(%8)
  %14 = Base.getindex(%13, 1)
  %15 = Base.getindex(%13, 2)
  %16 = Base.getindex(%13, 3)
  %17 = %12 + %16
  %18 = Base.tuple(0, %15, %17)
  return %18
```

and it calculates the gradient with respect to the input as
```julia
julia> pb(1.0)
(0, 1.0, 1.5403023058681398)
```
where the first item is gradient with parameters of the function itself.

## Conclusion
The above examples served to demonstrate that `@generated` functions offers extremely powerful paradigm, especially if coupled with manipulation of intermediate representation. Within few lines of code, we have implemented reasonably powerful profiler and reverse AD engine. Importantly, it has been done without a single-purpose engine or tooling. 
  