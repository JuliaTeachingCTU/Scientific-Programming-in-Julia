# Manipulating Intermediate Represenation (IR)

```@setup lec09
using InteractiveUtils: @code_typed, @code_lowered, code_lowered
```

## Generated functions
Sometimes, especially as we will see later, it is handy to generate function once the types with which the function is called are known. For example if we have function `foo(args...)`, we can generate different body for different length of `Tuple` and types in `args`. Wait, do we need generated functions for this? Not really, as 
- we can deal with variability of `args` using normal control-flow logic `if length(args) == 1 elseif ...`
- we can (automatically) generate (a possibly infinite) set of functions `foo` specialized for each length of args (or combination of types of args) and let multiple dispatch to deal with this
- we cannot deal with this situation with macros, because macros do not see types, only parsed AST, which is in this case always the same.

Generated functions elegantly deals with that situation, as we can specialize the code for a given type of argumnets. Generated functions, like macros, therefore **return expressions** and not **results**. But unlike macros, they do not have access to values of arguments, but to their types (the arguments are of type `Type`). They are also called when compiler needs (which means at least once for each combination of arguments, but possibly more times due to code invalidation).

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
which shows that the body of `genplus` is called for each combination of types of parameters, but the generated code is called whenever `genplus` is called.

Generated functions has to be pure in the sense that they are not allowed to have side effects, for example modifying some global variables. Note that printing is not allowed in pure functions, as it modifies the global buffer. This rule is not strict, including things, but not obeying it can lead to unexpected errors, as you do not know, at which moment the functions will be called.

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
@code_typed unrolled_map(+, x, y)
```
and compare this to the code generated by the non-generated version `permuted_map`:
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
It is expanded into a function with an if-condition, where the first branch `$(Expr(:generated))` generates the expression `:(x + x)` and returns it. The other spits out an error saying that the function has only a generated version. This suggests the possibility (and reality) to implement two versions of the same function; A generated and a *normal* version. It is left up to the compiler to decide which one to use. It is entirely up to the author to ensure that both versions are the same. Which version will the compiler take? The last comment on [23168](https://github.com/JuliaLang/julia/pull/23168) (as of time of writing) states:

"*Currently the `@generated` branch is always used. In the future, which branch is used will mostly depend on whether the JIT compiler is enabled and available, and if it's not available, then it will depend on how much we were able to compile before the compiler was taken away. So I think it will mostly be a concern for those that might need static compilation and JIT-less deployment.*"

## Contextual dispatch / overdubbing
Imagine that under some circumstances (context), you would like to use alternative implementations of some functions. One of the most cited motivations for this is automatic differentiation, where you would like to take the code **as-is** and calculate gradients with respect to some variables. Other use cases of this approach are mentioned in `Cassette.jl`:

"*Downstream applications for Cassette include dynamic code analysis (e.g. profiling, rr-style debugging, etc.), JIT compilation to new hardware/software backends, automatic differentiation, interval constraint programming, automatic parallelization/rescheduling, automatic memoization, lightweight multistage programming, graph extraction, and more.*"

In theory, we can do all the above by directly modifying the code or introducing new types, but that may require a lot of coding and changing of foreign libraries.

The technique we desire is called contextual dispatch, which means that under some context, we invoke a different function. The library `Casette.jl` provides a high-level API for overdubbing, but it is interesting to see, how it works, as it shows, how we can "interact" with the lowered code before the code is typed.

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
The lowered form is very nice, because on the left hand, there is **always** one variable and the right-hand side is simplified to have (mostly) a single call / expression. Moreover, in the lowered form, all control flow operations like `if`, `for`, `while` and exceptions are converted to `Goto` and `GotoIfNot`, which simplifies their handling. 

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
In overdubbing, our intention is to recursively dive into called function definitions and modify / change their code. In our example below, with which we will demonstrate the manual implementation (for educational purposes), our goal is to enclose each function call with statements that log the exection time. This means we would like to implement a simplified recording profiler. This functionality cannot be implemented by a macros, since macros do not allow us to dive into function definitions. For example, in our function `foo`, we would would not be able to dive into the definition of `sin` (not that this is a terribly good idea, but the point should be clear).

The overdubbing pattern works as follows.
1. We define a `@generated function overdub(f, args...)` which takes as a first argument a function `f` and then its arguments.
2. In the function `overdub` we retrieve the `CodeInfo` for `f(args...)`, which is possible as we know types of the arguments at this time.
3. We modify the the `CodeInfo` of `f(args...)` according to our liking. Importantly, we replace all function calls `some_fun(some_args...)` with `overdub(some_fun, some_args...)` which establishes the recursive pattern.
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
Above, we also filled the `slotflags`. Authors admit that names `:f` and `:args` in the above should be replaced by a `gensym`ed name, but they do not anticipate this code to be used for some bigger problems where name-clashes might occur.
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
    ### Retrieving the code properly

    Consider a following function:
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

## Implementing the profiler with IRTools
The above implementation of the profiler has shown, that rewriting the IR code manually is doable, but requires a lot of careful bookkeeping. `IRTools.jl` makes our life much simpler, as they take away all the needed book-keeping and let us focus on what is needed.

```julia
function foo(x, y)
   z =  x * y
   z + sin(y)
end
ir = @code_ir foo(1.0, 1.0)
```
We can see that at first sight, the representation of the lowered code in IRTools is similar to that of `CodeInfo`. Some notable differences:
- `SlotNumber` are converted to `SSAValues`
- SSA form is divided into blocks by `GotoNode` and `GotoIfNot` in the parsed `CodeInfo`
- SSAValues do not need to be ordered. The reordering is deffered to the moment when one converts `IRTools.Inner.IR`  back to the `CodeInfo`.

Let's now use the IRTools to insert the timing statements into the code for `foo` as 
```julia
ir = @code_ir foo(1.0, 1.0)
for b in IRTools.blocks(ir)
    for (v, ex) in b
        if timable(ex.expr)
            fname = exportname(ex.expr)
            insert!(b, v, xcall(Main, :record_start, fname))
            insertafter!(b, v, xcall(Main, :record_end, fname))
        end
    end
end

julia> ir
1: (%1, %2, %3)
  %7 = Main.record_start(:*)
  %4 = %2 * %3
  %8 = Main.record_end(:*)
  %9 = Main.record_start(:sin)
  %5 = Main.sin(%3)
  %10 = Main.record_end(:sin)
  %11 = Main.record_start(:+)
  %6 = %4 + %5
  %12 = Main.record_end(:+)
  return %6
```
Observe that the statements are on the right places but they are not ordered.
We can turn the `ir` object into and anonymous function as
```julia
f = func(ir)
reset!(to)
f(nothing, 1.0, 1.0)
to
```
where we can observe that our profiler was working as it should. But this is not yet our final goal. Originally, our goal was the profiler to recursivelly dive into the nested functions. IRTools offers macro `@dynamo`, which is similar to `@generated` but simplifies our job by allowing to return the `IRTools.Inner.IR` object and it also takes care of properly renaming the arguments. With that we write
```

profile_fun(f::Core.IntrinsicFunction, args...) = f(args...)
profile_fun(f::Core.Builtin, args...) = f(args...)

@dynamo function profile_fun(f, args...)
    ir = IRTools.Inner.IR(f, args...)
    for b in IRTools.blocks(ir)
        for (v, ex) in b
            if timable(ex.expr)
                fname = exportname(ex.expr)
                insert!(b, v, xcall(Main, :record_start, fname))
                insertafter!(b, v, xcall(Main, :record_end, fname))
            end
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
Also, the first two definitions of `profile_fun` for `Core.IntrinsicFunction` and for `Core.Builtin` prevent trying to dive into functions which does not have an ir representation. And that's all. The full code is 
```julia
using IRTools
using IRTools: var, xcall, insert!, insertafter!, func, recurse!, @dynamo
include("calls.jl")
resize!(to, 10000)

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
    for b in IRTools.blocks(ir)
        for (v, ex) in b
            if timable(ex.expr)
                fname = exportname(ex.expr)
                insert!(b, v, xcall(Main, :record_start, fname))
                insertafter!(b, v, xcall(Main, :record_end, fname))
            end
        end
    end
    for (x, st) in ir
        recursable(st.expr) || continue
        ir[x] = xcall(profile_fun, st.expr.args...)
    end
    # recurse!(ir)
    return ir
end
reset!(to)
profile_fun(foo, 1.0, 1.0)
to
```
where you should notice the long time the first execution of `profile_fun(foo, 1.0, 1.0)` takes. This is caused by the compiler specializing for every function into which we dive into. The second execution of `profile_fun(foo, 1.0, 1.0)` is fast. It is also interesting to observe how the time of the compilation is logged by the profiler. The output of the profiler `to` is not shown here due to the length of the output.

## Petite zygote

### some thoughts
ri is the variable
∂i is a function to which I need to pass the ∂ri to get the gradinets 

# TODO

* show constant gradient of linear function in LLVM
