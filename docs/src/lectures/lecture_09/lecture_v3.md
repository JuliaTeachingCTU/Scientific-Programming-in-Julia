# Source-to-Source Automatic Differentiation

This lecture is touching internals of julia compiler, which are not part of the official API and therefore are susceptible to break. The text was tested with Julia version 1.13
```julia
julia> versioninfo()
Julia Version 1.13.3
Commit 67ac47b9cd8 (2026-10-25 15:28 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: macOS (arm64-apple-darwin27.0.0)
  CPU: 14 × Apple M4 Pro
  WORD_SIZE: 64
  LLVM: libLLVM-20.1.8 (ORCJIT, apple-m4)
  GC: Built with stock GC
Threads: 1 default, 1 interactive, 1 GC (on 14 virtual cores)
```

Before diving into the next adventure in automatic differentiation (AD), we spent some time exploring the role of *rules* in AD.

The automatic differention libraries consists of two parts.
1. The engine which performs the differentiation (think about chainrule),
2. rules teaching the engine how to differentiate basic functions.

This is not very different to how people are differentiating by hand. They know how to differentiate basic functions (`sin`, `exp`, `+`, `/`, etc.) and they apply chainrule to differentiate complex functions. AD engine therefore either know, how to differentiate function because it has a rule for it, or it recursively decomposes the function to simple function it can differentiate. Functions which are differentiated automatically have the same signature and functionality as rules provided by the user and therefore they can be used as new rules.

In theory, it should be sufficient to define "few" basic rules and the rest should be handled by the AD system. The practice is different and there is a value of *strategically* defining custom rules. The reasons are
1. Performance. AD system has non-trivial overhead and therefore it does not make sense to differentiate operations like matrix multiplication.
2. Memory: Definining custom rules can save a lot of memory.
3. Numerical stability: Defining custom rules can improve numerical stability.
4. Approximate gradients: Sometimes, we need to differentiate function which is non-differentiable, but it has a differentiable approximation.

## ChainRules.jl
Julia has relatively large number of AD systems. At some moment, Katherine Frames White has decided that it might make sense to unify the rule system for all systems, such that AD systems can focus on the engine, which lead to creation of [ChainRules.jl](https://juliadiff.org/ChainRules.jl). ChainRules.jl is a package which provides a common interface for defining rules. The project was quite successful, though it has fall short of its intention, as some AD systems ([Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl), [Mooncake.jl](https://github.com/compintell/Mooncake.jl)) requires their own special rules, but frequently they can use rules defined by Chainrules.jl. The Chainrules had an impact on the ecosystem, as it discovered problems that have not been thought of before.

The rules are defined as a function which takes the function to differentiate and the arguments of the function and returns the gradient with respect to the arguments. The gradient is returned as a tuple of the same length as the number of arguments of the function. The gradient can be either a number or a special type `NoTangent()` which indicates that the gradient is not defined. The rules can be defined for any function, including user defined functions.

Let's investigate an example taken from the [ChainRules.jl documentation.](https://juliadiff.org/ChainRulesCore.jl/dev/rule_author/example.html) 
There is  a struct
```julia
struct Foo{T}
    A::Matrix{T}
    c::Float64
end
```
for which we define custom multiplication rule
```julia
function foo_mul(foo::Foo, b::AbstractArray)
    return foo.A * b
end
```
The rule for reverse AD, `rrule` is defined as 
```julia
function ChainRulesCore.rrule(::typeof(foo_mul), foo::Foo{T}, b::AbstractArray) where T
    y = foo_mul(foo, b)
    function foo_mul_pullback(ȳ)
        f̄ = NoTangent()
        f̄oo = Tangent{Foo{T}}(; A=ȳ * b', c=ZeroTangent())
        b̄ = @thunk(foo.A' * ȳ)
        return f̄, f̄oo, b̄
    end
    return y, foo_mul_pullback
end
```
- The first interesting thing is the signature of an `rrule`, which as a first argument contains the type of the function (think about it as an unique identifier of the function) and its argument. The concrete `rrule` is selected through multiple dispatch and there can be many variants for the same function. Also, if you list all methods `methods(rrule),` you will see number of rrules currently at the system.
- `rrule` returns the output of the function to be differentiated, and a function computing the gradient with respect to its arguments. This function is called *pullback*. Its input is the gradient with respect to the output of the function. Its output is the gradient with respect to function arguments. Note that since the function itself can have parameters, the first return value of the pullback is the gradient with respect to function parameters.
- The gradient can be either numerical gradient, or it might not exist (signaled by `NoTangent()`), or it can be zero, which is signaled by `ZeroTangent()`.
- The gradient with respect to the `struct` is returned as a `Tangent` type with type equal to the `struct`. The `Tangent` type contains `NamedTuple` with fields storing individual gradients. Type signature ensures we are adding right things.
- The `@thunk` macro is used for delayed evaluation.
- The `pullback` is in fact closure, which allows to communicate values from forward pass to the backward pass.
- the `rrule` is evaluated during the forward pass, `pullbacks` are evaluated during the reverse pass.

```julia
using Zygote
x = Foo(rand(2,2), 1.0)
y = rand(2,10)
gradient(x -> sum(foo_mul(x,y)), x)
```

Since *pullback* is a closure, it is used to communicate from forward to
reverse. For example when taking gradient through a sigmoid `σ(x) = inv(1 + exp(-x))`, the forward pass computes `y = σ(x)` and the backward pass computes `∇x = y * (1-y) * ∇y`. The `rrule` will therefore look like

```julia
σ(x) = inv(1 + exp(-x))         # numerically unstable

function σ(x)                   # numerically stable
    a = one(x) / (one(x) + exp(-abs(x)))
    return(x > 0 ? a : one(x) - a)
end

function ChainRulesCore.rrule(::typeof(σ), x::Real)
    y = σ(x)
    function σ_pullback(ȳ)
        f̄ = NoTangent()
        x̄ = y * (1-y)*ȳ
        return f̄, x̄
    end
    return y, σ_pullback
end
```

If we obtain the pullback as `y, pullback = rrule(σ, 1.0),` we can see that pullback contains `y` as field (it is closed in it as) `y == pullback.y.` The use of rrule will be slightly faster, since it will compute `exp` just once, which is expensive.
Compare a gradient of a version for which we have defined a rule and a version for which we have not defined a rule.

```julia
using Zygote
σ2(x) = inv(1 + exp(-x))
@benchmark gradient(sin ∘ σ2, 1.0)
@benchmark gradient(sin ∘ σ, 1.0)
```
Note that we add since to prevent compiler to optimize the code to be equivalent. The difference between both executions is tiny, because the overhead of AD is non-trivial. You can compare it to `@benchmark gradient(σ, 1.0)` and also observe `@code_native(σ, 1.0)`


## Source-to-Source Automatic Differentiation with IRCode

Let's now use `rrules` to create a simple AD system limited to function without control flow (unlike our previous tape-based system), which rewrites the code to insert statements needed for differentiation. While some 


The overall idea behind the construction is as follows. We assume to be given a function which we want to differentiate. First, we will verify if there is an `rrule`. If yes, we return the `rrule.` If not, we will create a function (functor), which will behave like `rrule.` The code to transform will be *lowered and typed* code (further called IRCode). This has the advantage that the code is in static single assignment form, stripped from all syntactic sugar, and it is typed. We can also ask compiler to provide certain optimization like SROA, inlining, constant propagation and dead code elimination. This means that we can start with relatively lean code.

!!! note "History"
    The idea of transforming intermidiate representation, at least in the context of Julia, was due to Mike Innes, who developped Zygote. Zygote worked on untyped IR provided in CodeInfo (see previous version of this lecture). In some cases, Zygote delivered excellent performance and in some cases the performance was not so stunning. While still being used as a main AD system from Flux, it will likely be superceeded by Mooncake (transforming typed IR) or Enzyme (transforming LLVM code).

Recall that `rrule` function is a function with the same arguments, as the original
function, but the it returns evaluation of the function and the *pullback*.
This `rrule` will be generated autotomatically from `IRCode` of the function to differentiate as follows:
  1. In the forward, we will replace each function call with a call to `rrule`. Since `rrule` return tuple (the evaluation of the function and the pullback), we extract the output of the original function and store the pullback.
  2. We construct the pullback function. We will iterate over the pullbacks stored from the forward pass in reverse order. Each pullback returns gradient with respect to the arguments, which need to be correctly accumulate it.
 
This is the essence of a simple AD without any control flow. The rest is some plumbing.

## Simple AD without control flow

As always, we start by importing few libraries.

```julia
import Core.Compiler as CC
using ChainRules
using Core: SSAValue, GlobalRef, ReturnNode
```

Let' start by defining a simple function to differentate.

```julia
function foo(x,y)
  z = x * y
  z + sin(x)
end
```

The first step is to get IRCode of the function. We can do this by calling `code_ircode` from Base.
Note that when calling `Base.code_ircode` we need to provide the types of the arguments, not their values.
This is because the compiler does not care about values, but about types.

```julia
ir, _ = only(Base.code_ircode(foo, (Float64, Float64); optimize_until = "CC: COMPACT_1"))
```

We can ask for different level of optimization, but such a simple function, it will not make much difference (an excerpt from function `run_passes_ipo_safe` defined in `Compiler/ss2ir/optimize.jl` :
```julia
    @pass "CC: CONVERT"   ir = convert_to_ircode(ci, sv)
    @pass "CC: SLOT2REG"  ir = slot2reg(ir, ci, sv)
    @pass "CC: COMPACT_1" ir = compact!(ir)
    @pass "CC: INLINING"  ir = ssa_inlining_pass!(ir, sv.inlining, ci.propagate_inbounds)
    @pass "CC: COMPACT_2" ir = compact!(ir)
    @pass "CC: SROA"      ir = sroa_pass!(ir, sv.inlining)
    @pass "CC: ADCE"      (ir, made_changes) = adce_pass!(ir, sv.inlining)
    if made_changes
        @pass "CC: COMPACT_3" ir = compact!(ir, true)
    end
    if is_asserts()
        @zone "CC: VERIFY_3" begin
            verify_ir(ir, true, false, optimizer_lattice(sv.inlining.interp), sv.linfo)
            verify_linetable(ir.debuginfo, length(ir.stmts))
        end
    end
    @label __done__  # used by @pass
```


The returned code will look like this
```julia
2 1 ─ %1 =   dynamic (_2 * _3)::Float64
3 │   %2 =   dynamic Main.sin(_2)::Float64
  │   %3 =   dynamic (%1 + %2)::Float64
  └──      return %3
   => Float64
```

The code is stored in `IRCode` data structure.

!!! note "IRCode*"
    ```julia
    struct IRCode
        stmts::InstructionStream
        argtypes::Vector{Any}
        sptypes::Vector{VarState}
        debuginfo::DebugInfoStream
        cfg::CFG
        new_nodes::NewNodeStream
        meta::Vector{Expr}
        valid_worlds::WorldRange
    end
    ```
    
    where
    * `stmts` is a stream of instruction (more in this below)
    * `argtypes` holds types of arguments of the function whose `IRCode` we have obtained (look to `ir.argtypes`)
    * `sptypes` is a vector of `VarState`. It contains information about parameters of types. Whenever the method signature uses `T`, this `T`  will become an element of `sptype` array, such that it is accessible by the compiler. The definition of `VarState` is 
    ```julia
    struct VarState
          typ          # lattice element: Const(Int64), a Type, Any, …
          undef::Bool  # true if the variable might not be defined at run time
      end
    ``` 
    and can be build by  `CC.sptypes_from_meth_instance(mi)` from the method's signature and the concrete specTypes it is being compiled for.
    * `debuginfo` holds information about code provenance, i.e. from which line of which method (and through which chain of inlined calls) each statement comes from. Julia 1.12 has reworked this block
    * `cfg` holds control flow graph, which contains building blocks and jumps between them
    * `new_nodes` is an infrastructure that can be used to insert new instructions to the existing `IRCode` . The idea behind is that since insertion requires a renumbering all statements, they are put in a separate queue. They are put to correct position with a correct `SSANumber`  by calling `compact!`.
    * `meta` holds `Expr(:meta, ...)` annotations of the function (e.g. `@inline`).
    * `valid_worlds` is the range of world ages in which the code is valid (if you redefine a method called from the function, the `IRCode` becomes stale).

    **InstructionStream**

    ```julia
    struct InstructionStream
        stmt::Vector{Any}
        type::Vector{Any}
        info::Vector{CallInfo}
        line::Vector{Int32}
        flag::Vector{UInt32}
    end
    ```
    where
    * `stmt` is a vector of instructions, stored as `Expr`essions. The allowed fields in `head` are described [here](https://docs.julialang.org/en/v1/devdocs/ast/#Expr-types)
    * `type` is the type of the value returned by the corresponding statement
    * `info` holds what the type inference has learned about the call, e.g. `MethodMatchInfo` contains the methods that can be called. It is used by the inliner. When we create statements ourselves, we set it to `NoCallInfo()`.
    * `line` holds the debug information of the statements. Every statement has three `Int32` entries** (since Julia 1.12), therefore `length(ir.stmts.line) == 3*length(ir.stmts)` and the information about the `i`-th statement is in `ir.stmts.line[3i-2:3i]`. The first number is an index into `ir.debuginfo.linetable` (the debuginfo of the lowered code, which maps to line numbers of the source), the second is an index into `ir.debuginfo.edges` (debuginfo of inlined functions, `0` if the statement was not inlined), and the third is an index into the line table of that edge. Note that `ir.debuginfo.codelocs` is the very same vector (`ir.debuginfo.codelocs === ir.stmts.line`), which is an invariant we need to keep when constructing `IRCode` by hand.
    * `flag`  are some flags providing additional information about the statement (see `IR_FLAG_*` constants in `Core.Compiler`).
      - `1 << 0` = `IR_FLAG_INBOUNDS` statement is marked as `@inbounds`
      - `1 << 1` = `IR_FLAG_INLINE` statement is marked as `@inline`
      - `1 << 2` = `IR_FLAG_NOINLINE` statement is marked as `@noinline`
      - `1 << 3` = `IR_FLAG_CONSISTENT` statement returns the same value for the same inputs
      - `1 << 4` = `IR_FLAG_EFFECT_FREE` statement does not have side effects
      - `1 << 5` = `IR_FLAG_NOTHROW` statement never throws
      - ... 
      - `1 << 16` = `IR_FLAG_REFINED` statement should be (re)inferred by the type inference
      - `1 << 17` = `IR_FLAG_UNUSED` the result of the statement is not used

    For the above `foo` function, the InstructionStream looks like

    ```julia
    julia> for i in 1:length(ir.stmts)
               s = ir.stmts[i]
               println(s[:flag], " | ",  s[:stmt], " | ", s[:type], " | ", ir.stmts.line[3i-2:3i])
           end
    9336   | _2 * _3      | Float64 | Int32[2, 0, 0]
    9304   | Main.sin(_2) | Float64 | Int32[6, 0, 0]
    9336   | %1 + %2      | Float64 | Int32[7, 0, 0]
    131072 | return %3    | Any     | Int32[8, 0, 0]
    ```
    We can index into the statements as `ir.stmts[1]`, which provides a "view" into the vector. To obtain the first instruction, we can do `ir.stmts[1][:stmt]` (before Julia 1.11, the field was called `:inst`).

!!! warning "Julia 1.14"
    Starting with Julia 1.14, the lowering puts the called function into a separate statement. The `IRCode` of `foo` therefore looks like
    ```julia
    2 1 ─ %1 = Main.:*::Core.Const(*)
      │   %2 =   dynamic (%1)(_2, _3)::Float64
    3 │   %3 = Main.:+::Core.Const(+)
      │   %4 = Main.sin::Core.Const(sin)
      │   %5 =   dynamic (%4)(_2)::Float64
      │   %6 =   dynamic (%3)(%2, %5)::Float64
      └──      return %6
    ```
    The statements `%1`, `%3`, and `%4` are `GlobalRef`s. Our transformation below has to take them into account.


### Implementing forward pass
Let's now go back to the problem of automatic differentiation. Recall the IRCode of the
`foo` function looks like this
```julia
2 1 ─ %1 =   dynamic (_2 * _3)::Float64
3 │   %2 =   dynamic Main.sin(_2)::Float64
  │   %3 =   dynamic (%1 + %2)::Float64
  └──      return %3
   => Float64
```
The forward part needs to replace each call of the function by a call to `rrule` and store pullbacks.
So in pseudocode, we want something like
```julia
(%1, %2) = rrule(*, _2, _3)
(%3, %4) = rrule(Main.sin, _2)
(%5, %6) = rrule(Base._, %1, %3)
return(%5, tuple(%2, %4,%6))
```
In the above pseudocode, `%1, %3, %5` are ouputs of function `*, sin, +` respectively, and `%2, %4, %6`
are their pullbacks. The function therefore return the correct value and information for the pullback.
The above pseudocode is not a valid IRCode, since SSA to assign only one variable.

To implement the code performing the above transformation, we initiate few variables


```julia
adinfo = []             # storage for informations about pullbacks, needed for the construction of the reverse pass
new_insts = Any[]       # storate for instructions
new_line =  Int32[]     # debuginfo of new instructions (three Int32 per instruction)
ssamap = Dict{SSAValue,Any}() # this maps old SSA values to new SSA values, since they need to be linearly ordered.
```

We also define a remap_ssa function which will be used to map old SSA values to new SSA values.

```julia
remap_ssa(d, args::Tuple) = map(a -> remap_ssa(d,a), args)
remap_ssa(d, args::Vector) = map(a -> remap_ssa(d,a), args)
remap_ssa(d, r::ReturnNode) = ReturnNode(remap_ssa(d, r.val))
remap_ssa(d, x::SSAValue) = d[x]
remap_ssa(d, x) = x

struct PullStore{T<:Tuple}
  data::T
end

PullStore(args...) = PullStore(tuple(args...))
Base.getindex(p::PullStore, i) = p.data[i]
```

```julia
lineinfo(ir, i) = ir.stmts.line[3i-2:3i]
```

The main loop transforming the function looks like foollows

```julia
for (i, stmt) in enumerate(ir.stmts)
    inst = stmt[:stmt]
    line = lineinfo(ir, i)
    if inst isa Expr && inst.head == :call
        new_inst = Expr(:call, GlobalRef(ChainRules, :rrule), remap_ssa(ssamap, inst.args)...)
        push!(new_insts, new_inst)
        append!(new_line, line)
        rrule_ssa = SSAValue(length(new_insts))

        push!(new_insts, Expr(:call, GlobalRef(Base, :getindex), rrule_ssa, 1))
        append!(new_line, line)
        val_ssa = SSAValue(length(new_insts))
        ssamap[SSAValue(i)] = val_ssa

        push!(new_insts, Expr(:call, GlobalRef(Base, :getindex), rrule_ssa, 2))
        pullback_ssa = SSAValue(length(new_insts))
        append!(new_line, line)
        push!(adinfo, (;old_ssa = i, inst = inst, pullback_ssa))
        continue
    end

    if inst isa ReturnNode
        push!(new_insts, Expr(:call, GlobalRef(Main, :PullStore), map(x -> x[end], adinfo)...))
        pullback_ssa = SSAValue(length(new_insts))
        append!(new_line, line)

        push!(new_insts, Expr(:call, GlobalRef(Base, :tuple), remap_ssa(ssamap, inst.val), pullback_ssa))
        returned_tuple = SSAValue(length(new_insts))
        append!(new_line, line)

        push!(new_insts, ReturnNode(returned_tuple))
        append!(new_line, line)
        continue
    end

    # Since Julia 1.14, the called function is a separate statement, e.g. `%1 = Main.:*`.
    # We do not emit it, but we substitute it directly into the call.
    if inst isa GlobalRef
        ssamap[SSAValue(i)] = inst
        continue
    end
    error("unknown node $(i)")
end
```

After the executing the loop, we obtain stream of new instructions in the `new_insts` and information about how to construct
the reverse pass stored in `adinfo`.
Now, we need to consruct valid IRCode, which can be executed. We do this by first
constructing Instruction stream as

```julia
stmts = CC.InstructionStream(
   new_insts,
   Any[Any for _ in new_insts],
   CC.CallInfo[CC.NoCallInfo() for _ in new_insts],
   new_line,
   fill(CC.IR_FLAG_REFINED, length(new_insts)),
)
```

where (i) we have marked all instruction to be (re)inferred by the type inference (`CC.IR_FLAG_REFINED`),
and we set all return type to `Any` to make them amenable for typing. Notice that the
constructor requires vectors of exactly types `Vector{Any}` and `Vector{CallInfo}`
(`fill(Any, n)` would create `Vector{DataType}`).

From this stream, we can construct new IRCode. Since Julia 1.12, the `IRCode` does not hold
the `linetable` but `DebugInfoStream`. We reuse the debuginfo of the original function (which
gives meaning to the numbers in `new_line`), but we need to replace `codelocs`
with our new line information, since `debuginfo.codelocs` and `stmts.line` have to be the same vector.

```julia
cfg = CC.compute_basic_blocks(new_insts)
debuginfo = CC.copy(ir.debuginfo)
debuginfo.codelocs = new_line
forward_ir = CC.IRCode(stmts, cfg, debuginfo, Any[Tuple{}, Float64, Float64], Expr[], CC.VarState[])
```

Since we have kept the debuginfo, the printed `forward_ir` shows from which line of `foo` each statement comes from
```julia
2 1 ─ %1  =   dynamic ChainRules.rrule(Main.:*, _2, _3)::Any
  │   %2  =   dynamic Base.getindex(%1, 1)::Any
  │   %3  =   dynamic Base.getindex(%1, 2)::Any
3 │   %4  =   dynamic ChainRules.rrule(Main.sin, _2)::Any
  │   %5  =   dynamic Base.getindex(%4, 1)::Any
  │   %6  =   dynamic Base.getindex(%4, 2)::Any
  │   %7  =   dynamic ChainRules.rrule(Main.:+, %2, %5)::Any
  │   %8  =   dynamic Base.getindex(%7, 1)::Any
  │   %9  =   dynamic Base.getindex(%7, 2)::Any
  │   %10 =   dynamic Main.PullStore(%3, %6, %9)::Any
  │   %11 =   dynamic Base.tuple(%8, %10)::Any
  └──       return %11
```

which can be executed by wrapping it into `OpaqueClosure`

```julia
oc = Core.OpaqueClosure(forward_ir)
value, pullback = oc(1.0, 1.0)
```

We can verify the value is equal to the output of the original function `foo`

```julia
value == foo(1.0, 1.0)
```

And the pullback contains functions computing gradient with respect to individual functions within `foo`

```julia
pullback[3](1.0) == (ChainRules.NoTangent(), 1.0, 1.0)
pullback[2](1.0) == (ChainRules.NoTangent(), cos(1.0))
pullback[1](1.0) == (ChainRules.NoTangent(), 1.0, 1.0)
```

!!! note "Type inference"
    
    So far the IRCode we have constructed was "untyped". We can use Julia's type inference
    to type the IRCode. Type type inference can be achieved through the following function,
    which was kindly provided to the community by Katherine Frames White and the actual versions
    were copied from Mooncake.jl.

    ```julia
    function infer_ir!(ir::CC.IRCode)
        return __infer_ir!(ir, CC.NativeInterpreter(), __get_toplevel_mi_from_ir(ir, Main))
    end
    ```

    Given some IR, generates a MethodInstance suitable for passing to infer_ir!, if you don't
    already have one with the right argument types. [Credit to@oxinabox:
    (https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802)

    ```julia
    function __get_toplevel_mi_from_ir(ir, _module::Module)
        mi = ccall(:jl_new_method_instance_uninit, Ref{Core.MethodInstance}, ());
        mi.specTypes = Tuple{map(CC.widenconst, ir.argtypes)...}
        mi.def = _module
        return mi
    end
    ```

    Run type inference and constant propagation on the ir. [Credit to @oxinabox:]
    (https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802#file-example-jl-L54)
    Since Julia 1.12, `MethodInfo` has been replaced by `SpecInfo` and `_ir_abstract_constant_propagation`
    has been renamed to `ir_abstract_constant_propagation`. Julia 1.14 has removed the `world` argument
    of `IRInterpretationState` (only `min_world` and `max_world` remain).

    ```julia
    function __infer_ir!(ir, interp::CC.AbstractInterpreter, mi::CC.MethodInstance)
        nargs = length(ir.argtypes) - 1
        spec_info = CC.SpecInfo(nargs, #=isva=#false, #=propagate_inbounds=#true, nothing)
        world = CC.get_inference_world(interp)
        @static if VERSION >= v"1.14-"
            irsv = CC.IRInterpretationState(interp, spec_info, ir, mi, ir.argtypes, world, world)
        else
            irsv = CC.IRInterpretationState(interp, spec_info, ir, mi, ir.argtypes, world, world, world)
        end
        CC.ir_abstract_constant_propagation(interp, irsv)
        return ir
    end
    ```

With that, we can create closure with type inference

```julia
toc = Core.OpaqueClosure(infer_ir!(forward_ir))
value, pullback = toc(1.0, 1.0)
```

### Helper functions
Before moving forward, we define we helper functions which would simplify the construction
if the IRCode from the stream of instructions. We will also automatically run type inference.

```julia
function ircode(
    insts::Vector{Any}, argtypes::Vector{Any}, sptypes::Vector{CC.VarState}=CC.VarState[]
)
    cfg = CC.compute_basic_blocks(insts)
    stmts = __insts_to_instruction_stream(insts)
    # debuginfo without any source information; it has to share the vector with `stmts.line`
    debuginfo = CC.DebugInfoStream(stmts.line)
    meta = Expr[]
    ir = CC.IRCode(stmts, cfg, debuginfo, argtypes, meta, sptypes)
    infer_ir!(ir)
end

function __insts_to_instruction_stream(insts::Vector{Any})
    n = length(insts)
    return CC.InstructionStream(
        insts,
        Any[Any for _ in 1:n],
        CC.CallInfo[CC.NoCallInfo() for _ in 1:n],
        zeros(Int32, 3n),   # three zeros per statement = no line information
        fill(CC.IR_FLAG_REFINED, n),
    )
end
```

With that, the forward function looks as follows

```julia
function construct_forward(ir)
    adinfo = []
    new_insts = Any[]
    ssamap = Dict{SSAValue,Any}()
    for (i, stmt) in enumerate(ir.stmts)
        inst = stmt[:stmt]
        if inst isa Expr && inst.head == :call
            new_inst = Expr(:call, GlobalRef(ChainRules, :rrule), remap_ssa(ssamap, inst.args)...)
            push!(new_insts, new_inst)
            rrule_ssa = SSAValue(length(new_insts))

            push!(new_insts, Expr(:call, GlobalRef(Base, :getindex), rrule_ssa, 1))
            val_ssa = SSAValue(length(new_insts))
            ssamap[SSAValue(i)] = val_ssa

            push!(new_insts, Expr(:call, GlobalRef(Base, :getindex), rrule_ssa, 2))
            pullback_ssa = SSAValue(length(new_insts))
            push!(adinfo, (;old_ssa = i, inst = inst, pullback_ssa))
            continue
        end

        if inst isa ReturnNode
            # Wrap all pullbacks to PullStore
            push!(new_insts, Expr(:call, GlobalRef(Main, :PullStore), map(x -> x[end], adinfo)...))
            pullback_ssa = SSAValue(length(new_insts))

            # construct the tuple (value, pullbacks) to return
            push!(new_insts, Expr(:call, GlobalRef(Base, :tuple), remap_ssa(ssamap, inst.val), pullback_ssa))
            returned_tuple = SSAValue(length(new_insts))

            push!(new_insts, ReturnNode(returned_tuple))
            continue
        end

        # Julia 1.14+: the called function is a separate statement
        if inst isa GlobalRef
            ssamap[SSAValue(i)] = inst
            continue
        end
        error("unknown node $(i)")
    end
    
    # Finally, we construct the valid IR code from new statements.

    argtypes = Any[Tuple{}, ir.argtypes[2:end]...]
    new_ir = ircode(new_insts, argtypes)
    (new_ir, adinfo)
end
```

We can try to run the above as 
```julia
forward_ir, adinfo = construct_forward(ir)
oc = Core.OpaqueClosure(forward_ir)
value, pullback = oc(1.0, 1.0)
```

### Implementing reverse pass
The reverse parts of a linear IR is relatively simple. In a nutshell, the code
code needs to iterate over the pullbacks in the reverse order, execute them, and
accumulate the gradients with respect to the arguments. The accumulation of the
gradient is the most difficult part here.

The function `construct_pullback` will have three arguments. First are the information
about id of arguments and return values of the original pullback, which we have saved
during construction of the forward part (not execution). The second argument are the types
of values returned by the forward part, which are useful for typing. The third argument is
the SSA value returned by the original function, which is the value whose gradient is the input
of the pullback. (Previous version of the lecture assumed it is `SSAValue(length(pullbacks))`, which is true
only if all statements are calls, and this is not the case since Julia 1.14.)

The function we construct will have two arguments. The first will be information
communicated from the forward part to the reverse, which contains the varibles closed
in pullbacks. The second argument will be the gradient with respect to the output of
the function we are derivating.

```julia
function construct_pullback(pullbacks, ::Type{<:Tuple{R,P}}, ret) where {R,P}
    diffmap = Dict{Any,Any}() # this will hold the mapping where is the gradient with respect to SSA.
    # the argument of the pullback we are defining is a gradient with respect to the returned value
    diffmap[ret] = Core.Argument(3)
    reverse_inst = []

    # now we iterate over pullbacks and execute one by one with correct argument
    for pull_id in reverse(axes(pullbacks,1))
        ssa_no, inst, _ = pullbacks[pull_id]
        # first we extract the pullback from a tuple of pullbacks
        push!(reverse_inst, Expr(:call, GlobalRef(Base, :getindex), Core.Argument(2), pull_id))

        #then we call the pullback with a correct argument
        push!(reverse_inst, Expr(:call, SSAValue(length(reverse_inst)), diffmap[SSAValue(ssa_no)]))
        arg_grad = SSAValue(length(reverse_inst))
        
        # then we extract gradients with respect to the argument of the instruction and record all the calls

        for (i, a) in enumerate(inst.args)
            i == 1 && continue # we omit gradient with respect to the name of the function and rrule
            if haskey(diffmap, a)  # we need to perform addition
                push!(reverse_inst, Expr(:call, GlobalRef(Base, :getindex), arg_grad, i))
                new_val = SSAValue(length(reverse_inst))
                old_val = diffmap[a]
                push!(reverse_inst, Expr(:call, GlobalRef(Base, :+), old_val, new_val))
                diffmap[a] = SSAValue(length(reverse_inst))
            else
                push!(reverse_inst, Expr(:call, GlobalRef(Base, :getindex), arg_grad, i))
                diffmap[a] = SSAValue(length(reverse_inst))
            end
        end
    end

    # we create a Tuple with return values

    ∇args = collect(filter(x -> x isa Core.Argument, keys(diffmap)) )
    sort!(∇args, by = x -> x.n)
    push!(reverse_inst, Expr(:call, GlobalRef(Base, :tuple), [diffmap[a] for a in ∇args]...))
    returned_tuple = SSAValue(length(reverse_inst))
    push!(reverse_inst, ReturnNode(returned_tuple))

    ircode(reverse_inst, Any[Tuple{},P, R])
end
```

Let's now put things together. The automatically generated rules will be
stored in a struct `CachedGrad` containing forward and backward passes.
We aditionally define a function `gradient` which will execute the
forward and backward pass and return a closure containing adinfo from the forward
pass. With this trick, the pullback returned to the user contains only

```julia
struct CachedGrad{F<:Core.OpaqueClosure, R<:Core.OpaqueClosure}
    foc::F
    roc::R
end

function gradient(f::CachedGrad, args...)
    v, adinfo = f.foc(args...)
    v, f.roc(adinfo, one(eltype(v)))
end

totype(x::DataType) = x
totype(x::Type) = x
totype(x) = typeof(x)

function CachedGrad(f, args...)
    args = tuple(map(totype, args)...)
    ir, _ = only(Base.code_ircode(f, args; optimize_until = "CC: COMPACT_1"))
    ret = only(filter(x -> x isa ReturnNode, ir.stmts.stmt)).val
    forward_ir, pullbacks = construct_forward(ir)
    rt = CC.compute_ir_rettype(forward_ir)
    reverse_ir = construct_pullback(pullbacks, rt, ret)

    foc = Core.OpaqueClosure(forward_ir; do_compile = true)
    roc = Core.OpaqueClosure(reverse_ir; do_compile = true)
    CachedGrad(foc, roc)
end

function foo(x,y)
  z = x * y
  z + sin(x)
end

bar(x) = 5 * x
```

Test as

```julia
 gradient(CachedGrad(foo, 1.0, 1.0), 1.0, 1.0)
 gradient(CachedGrad(bar, 1.0), 1.0)
```

[1] [Autodiff by G. Dalle](https://gdalle.github.io/JuliaOptimizationDays2024-AutoDiff/#/)