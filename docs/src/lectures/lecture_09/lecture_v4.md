# Source-to-Source Automatic Differentiation

This lecture is touching internals of julia compiler, which are not part of the official API and therefore are susceptible to break. The text was tested with Julia 1.13.4 and 1.14.0-beta1
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

Let's now use `rrules` defined in `ChainRulesCore.jl` to create a simple source-to-source AD system limited to functions without control flow (unlike our previous tape-based system).
Almost all source-to-source AD systems follows following steps:

    1. Obtain the code of the function to differentate. 
    2. Convert the code into the SSA representation.
    3. Transform it the code to perform the differentiation. 
    4. Return the updated code back for the execution. 

Steps 1, 2 and 4 are plumbing, the real work is done in step 3. 

```
┌───────────────┬────────────────────────────────────────┬──────────────────────┬───────────────────────────────────────┬───────────────────────────────────────────┐
│               │            1. get the code             │     2. into SSA      │             3. transform              │         4. run, cache, invalidate         │
├───────────────┼────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────┤
│               │ generated function _pullback; IRTools  │ IRTools converts it  │ adjoint: calls are replaced by        │ returns a CodeInfo with edges, so Julia   │
│ Zygote        │ reads the lowered (untyped) code       │ to its own SSA IR    │ _pullback calls, and the pullback is  │ caches it and invalidates it              │
│               │                                        │                      │ another generated function            │ automatically                             │
├───────────────┼────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────┤
│ Diffractor    │ generated function; retrieve_code_info │ by hand with         │ diffract_ir!; pullbacks are opaque    │ replace_code_newstyle! plus ci.edges;     │
│ stage 1       │  gets the lowered code                 │ construct_ssa! and   │ closures created inside the generated │ Julia caches and invalidates              │
│               │                                        │ compact!             │  code                                 │                                           │
├───────────────┼────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────┤
│               │ an explicit call to build_rrule; its   │ already done: typed, │ make_ad_stmts!; calls become          │ MistyClosures, a cache of its own keyed   │
│ Mooncake      │ own MooncakeInterpreter calls          │  optimized IRCode    │ LazyDerivedRule / DynamicDerivedRule  │ by world age; redefining a method means   │
│               │ typeinf_ircode                         │                      │                                       │ building a new rule                       │
├───────────────┼────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────┤
│               │ GPUCompiler compiles the function to   │ already done: LLVM   │ the Enzyme LLVM plugin differentiates │ the result is JIT-compiled and called     │
│ Enzyme        │ LLVM IR (with its own interpreter that │ IR is SSA            │  every function the code calls        │ through a thunk; the cache is keyed by    │
│               │  controls inlining)                    │                      │                                       │ world age                                 │
├───────────────┼────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────┤
│ JAX (for      │ runs the function on abstract values   │ the jaxpr is already │ jvp / transpose on the jaxpr          │ lowers it to XLA and compiles; cached by  │
│ comparison)   │ to record a "jaxpr"                    │  SSA-like            │                                       │ input shapes and types                    │
├───────────────┼────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────┤
│ this lecture  │ Base.code_ircode                       │ already done: typed  │ forward_ir / reverse_ir               │ OpaqueClosure; cached in a Dict by        │
│               │                                        │ IRCode               │                                       │ signature; no invalidation                │
└───────────────┴────────────────────────────────────────┴──────────────────────┴───────────────────────────────────────┴───────────────────────────────────────────┘

```

In our example, the we will create a function which behaves like its `rrule`, i.e. it has the same arguments as the original function, and it returns the value of the function and the *pullback*. This new function is generated automatically from the *lowered and typed* code of the original function (further called IRCode). IRCode has the advantage that the code is in static single assignment form, stripped from all syntactic sugar, and it is typed.

The generated `rrule` works as follows:
  1. In the **forward pass**, every function call is replaced by a call to its `rrule`. Since `rrule` returns a tuple (the value of the function and its pullback), we extract the value of the original function and keep the pullback.
  2. At the end of the forward pass, we create the **pullback of the whole function** as a *closure*, which captures the pullbacks of the individual calls. This is exactly what the hand-written `rrule`s above do: `σ_pullback` is a closure capturing `y`.
  3. The body of this closure is the **reverse pass**. It calls the captured pullbacks in reverse order and accumulates the gradients with respect to the arguments.

The closure will be an *opaque closure*, which is a closure whose code we can generate ourselves. This is the essence of a simple AD without any control flow. The rest is plumbing.

!!! note "History"
    The idea of transforming intermediate representation, at least in the context of Julia, is due to Mike Innes, who developed Zygote. Zygote works on the untyped IR provided in `CodeInfo` (see the previous version of this lecture). In some cases, Zygote delivered excellent performance, and in some cases the performance was not so stunning. While still being the main AD system of Flux, it will likely be superseded by Mooncake (transforming typed IR) or Enzyme (transforming LLVM code). Creating the pullback as an opaque closure inside the transformed code is taken from [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl), an AD system written by Keno Fischer, which is no longer developed.

## Simple AD without control flow

As always, we start by importing few libraries.

```julia
import Core.Compiler as CC
using ChainRules
using Core: SSAValue, Argument, GlobalRef, ReturnNode
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

### What we want to generate
Let's now go back to the problem of automatic differentiation. Recall that the IRCode of `foo` looks like this
```
2 1 ─ %1 =   dynamic (_2 * _3)::Float64
3 │   %2 =   dynamic Main.sin(_2)::Float64
  │   %3 =   dynamic (%1 + %2)::Float64
  └──      return %3
```
The forward pass replaces each call by a call to `rrule` and keeps the pullbacks. At the end, it creates the pullback of `foo` as a closure capturing the three pullbacks. In pseudocode, we want something like
```
(%1, pb₁) = rrule(*, _2, _3)
(%2, pb₂) = rrule(sin, _2)
(%3, pb₃) = rrule(+, %1, %2)
pullback = Δ -> begin              # captures pb₁, pb₂, pb₃
    (_, ∂z, ∂sin) = pb₃(Δ)
    (_, ∂x)       = pb₂(∂sin)
    (_, ∂x′, ∂y)  = pb₁(∂z)
    (NoTangent(), ∂x + ∂x′, ∂y)
end
return (%3, pullback)
```
The question is how to create a closure whose body we generate ourselves.

### Opaque closures
Julia has a special kind of closure called an *opaque closure*. Like an ordinary closure, it is a function together with captured values. Unlike an ordinary closure, it is not a struct with a method defined for it, but an object holding its own code, and the compiler cannot inside it when it is called (hence *opaque*). For our AD system, there is one more important distiction, which is that `OpaqueClosure` can wrap any function. An ordinary closure's body must be written in the source, but  opaque closure's body can be created at the time of the construction, which is what we will use.

The experimental macro `Base.Experimental.@opaque` creates one:

```julia
function mk(y)
    z = sin(y)
    Base.Experimental.@opaque Δ -> Δ * z
end
```

```
julia> oc = mk(2.0)
(::Any)->◌::Any

julia> oc(3.0)
2.727892280477045

julia> oc.captures
(0.9092974268256817,)
```

The interesting part is how the closure looks in the IRCode of `mk`:

```
julia> ir_mk, _ = only(Base.code_ircode(mk, (Float64,); optimize_until = "CC: COMPACT_1"));

julia> ir_mk
3 1 ─ %1 =   dynamic Main.sin(_2)::Float64
4 │        nothing::Core.Const(Tuple{Any})
  │        nothing::Core.Const(Union{})
  │   %4 = $(Expr(:new_opaque_closure, Tuple{Any}, Union{}, :(Core.Any), true, (::Any)(Δ) @ Main REPL[1]:4, :(%1)))::Core.PartialOpaque(...)
  └──      return %4
```

The closure is created by a single statement `Expr(:new_opaque_closure, ...)` with the following arguments:
1. `Tuple{Any}` is the type of the arguments of the closure,
2. `Union{}` and `Any` are the lower and upper bounds of its return type,
3. `true` allows the compiler to infer the closure (it is always `true` here),
4. a `Method` holding the **code of the closure**,
5. the remaining arguments are the **captured values** (here `%1`, which is `z`).

The `Method` holds the *untyped* (lowered) code of the body of the closure:
```
julia> i = findfirst(s -> Meta.isexpr(s, :new_opaque_closure), ir_mk.stmts.stmt);

julia> Base.uncompressed_ast(ir_mk.stmts.stmt[i].args[5])
CodeInfo(
1 ─ %1 = Main.:*
│   %2 =   builtin Core.getfield(#self#, 1)
│   %3 =   dynamic (%1)(Δ, %2)
└──      return %3
)
```
The body has two arguments, `#self#` (the closure itself) and `Δ`. The captured values are read as `getfield(#self#, i)`.

This gives us the plan. We will
1. generate the code of the reverse pass, the body of the closure, and turn it into a `Method`;
2. generate the code of the forward pass, which ends with `Expr(:new_opaque_closure, ..., method, pb₁, pb₂, pb₃)`;
3. turn the forward pass into an opaque closure, which we can call.

Both passes are generated as *untyped* code (`CodeInfo`), and Julia's type inference types them when they are compiled. We cannot use the IRCode with type inference from the previous version of this lecture, since the inference on IRCode (the IR interpreter) does not understand `:new_opaque_closure`. In return, we do not need any of the plumbing needed to run type inference on IRCode. We still use the *typed* IRCode of the original function to learn what to transform.

### Untyped code
`CodeInfo` holds the lowered code of a function. We create one from a vector of statements with the following helper:

```julia
function lowered_codeinfo(insts::Vector{Any}, lines::Vector{Int32}, debuginfo, slotnames::Vector{Symbol})
    n = length(insts)
    ci = ccall(:jl_new_code_info_uninit, Ref{Core.CodeInfo}, ())
    ci.code = insts
    ci.ssavaluetypes = n                  # the code is not inferred
    ci.ssaflags = zeros(UInt32, n)
    ci.slotnames = slotnames              # names of the arguments, the first one is #self#
    ci.slotflags = zeros(UInt8, length(slotnames))
    ci.nargs = length(slotnames)
    ci.isva = false
    di = CC.copy(debuginfo)               # see debuginfo.md
    di.codelocs = lines
    ci.debuginfo = Core.DebugInfo(di, n)  # compress
    return ci
end
```

`ssavaluetypes` is the number of statements, which says that the code has not been inferred yet. The arguments are referred to as `Argument(i)`, where `Argument(1)` is `#self#`, as in IRCode. For the debuginfo, we reuse the debuginfo of the original function with new line information (three `Int32` per statement), such that the generated code points to the lines of the original function. See [debuginfo.md](debuginfo.md) for details.

### Implementing the forward pass
The forward pass goes over the statements of the original function. Every call is replaced by three statements: the call to `rrule` and the extraction of the value and of the pullback. We keep a map `ssamap` from the SSA values of the original code to the SSA values of the new code. For every call, we record into `adinfo` what is needed to construct the reverse pass: the index of the statement, the statement itself, the SSA value holding its pullback, and its line information.

```julia
lineinfo(ir, i) = ir.stmts.line[3i-2:3i]

remap_ssa(d, x::SSAValue) = d[x]
remap_ssa(d, x) = x

function construct_forward(ir; rule = GlobalRef(ChainRules, :rrule))
    length(ir.cfg.blocks) == 1 || error("control flow is not supported")
    adinfo = []
    insts, lines = Any[], Int32[]
    ssamap = Dict{SSAValue,Any}()
    ret = nothing
    for (i, stmt) in enumerate(ir.stmts)
        inst = stmt[:stmt]
        line = lineinfo(ir, i)
        if Meta.isexpr(inst, :call)
            push!(insts, Expr(:call, rule, [remap_ssa(ssamap, a) for a in inst.args]...))
            append!(lines, line)
            r = SSAValue(length(insts))

            push!(insts, Expr(:call, GlobalRef(Base, :getindex), r, 1))
            append!(lines, line)
            ssamap[SSAValue(i)] = SSAValue(length(insts))

            push!(insts, Expr(:call, GlobalRef(Base, :getindex), r, 2))
            append!(lines, line)
            push!(adinfo, (; old_ssa = i, inst, pullback_ssa = SSAValue(length(insts)), line))
        elseif inst isa GlobalRef                 # Julia 1.14+: the called function is a separate statement
            ssamap[SSAValue(i)] = inst            # we substitute it directly into the call
        elseif inst isa ReturnNode
            ret = (; val = inst.val, newval = remap_ssa(ssamap, inst.val), line)
        else
            error("unsupported statement $inst")
        end
    end
    return insts, lines, adinfo, ret
end
```

The keyword argument `rule` is the function which is called instead of the original functions. We fix it to `ChainRules.rrule` for a moment, but later we will change it to support recursion.

The returned statements are not complete yet. The end of the forward pass, which creates the pullback closure and returns, is added once we have the reverse pass. For `foo`, the statements so far are

```
%1 = ChainRules.rrule(Main.:*, _2, _3)
%2 = Base.getindex(%1, 1)
%3 = Base.getindex(%1, 2)
%4 = ChainRules.rrule(Main.sin, _2)
%5 = Base.getindex(%4, 1)
%6 = Base.getindex(%4, 2)
%7 = ChainRules.rrule(Main.:+, %2, %5)
%8 = Base.getindex(%7, 1)
%9 = Base.getindex(%7, 2)
```

and the pullbacks are `%3`, `%6`, and `%9`.

### Implementing the reverse pass
The reverse pass is the body of the pullback closure. It has two arguments: `Argument(1)` is the closure itself, from which we read the `k`-th captured pullback as `getfield(Argument(1), k)`, and `Argument(2)` is the gradient `Δ` with respect to the output.

The reverse pass iterates over the pullbacks in reverse order. Each pullback is called with the gradient of the output of its call, and it returns the gradients with respect to the arguments of the call. Pullbacks from ChainRules may return gradients as *thunks* (delayed computations, recall `@thunk` above), so we call `unthunk` on a gradient before passing it to the next pullback. For a plain number, `unthunk` does nothing. The gradients are accumulated in `diffmap`, which maps a value of the original code (an `SSAValue` or an `Argument`) to the SSA value of the reverse pass holding its gradient. If a value is used more than once (as `x` in `foo`), its gradients are summed.

At the end, the reverse pass returns the gradients with respect to the arguments of the original function, in the same format as the pullbacks from ChainRules: the first element is the gradient with respect to the function itself (`NoTangent()`). If an argument has not been used, its gradient is `ZeroTangent()`.

```julia
using ChainRules: NoTangent, ZeroTangent

function construct_pullback(adinfo, ret, nargs)
    insts, lines = Any[], Int32[]
    diffmap = Dict{Any,Any}(ret.val => Argument(2))     # Argument(2) = Δ
    for k in reverse(eachindex(adinfo))
        (; old_ssa, inst, line) = adinfo[k]
        push!(insts, Expr(:call, GlobalRef(Core, :getfield), Argument(1), k))   # k-th captured pullback
        append!(lines, line)
        pb = SSAValue(length(insts))

        push!(insts, Expr(:call, GlobalRef(ChainRules, :unthunk), diffmap[SSAValue(old_ssa)]))
        append!(lines, line)
        push!(insts, Expr(:call, pb, SSAValue(length(insts))))
        append!(lines, line)
        ∂ = SSAValue(length(insts))

        for (j, a) in enumerate(inst.args)
            j == 1 && continue        # gradient with respect to the called function
            push!(insts, Expr(:call, GlobalRef(Base, :getindex), ∂, j))
            append!(lines, line)
            if haskey(diffmap, a)     # the value has been used before, accumulate
                push!(insts, Expr(:call, GlobalRef(Base, :+), diffmap[a], SSAValue(length(insts))))
                append!(lines, line)
            end
            diffmap[a] = SSAValue(length(insts))
        end
    end
    ∇args = [get(diffmap, Argument(n), ChainRules.ZeroTangent()) for n in 2:nargs]
    push!(insts, Expr(:call, GlobalRef(Core, :tuple), ChainRules.NoTangent(), ∇args...))
    append!(lines, ret.line)
    push!(insts, ReturnNode(SSAValue(length(insts))))
    append!(lines, ret.line)
    return insts, lines
end
```

For `foo`, the reverse pass is
```
%1  = Core.getfield(_1, 3)          # pullback of +
%2  = ChainRules.unthunk(_2)
%3  = (%1)(%2)
%4  = Base.getindex(%3, 2)          # ∂z
%5  = Base.getindex(%3, 3)          # ∂sin
%6  = Core.getfield(_1, 2)          # pullback of sin
%7  = ChainRules.unthunk(%5)
%8  = (%6)(%7)
%9  = Base.getindex(%8, 2)          # ∂x from sin(x)
%10 = Core.getfield(_1, 1)          # pullback of *
%11 = ChainRules.unthunk(%4)
%12 = (%10)(%11)
%13 = Base.getindex(%12, 2)         # ∂x from x * y
%14 = %9 + %13                      # x is used twice
%15 = Base.getindex(%12, 3)         # ∂y
%16 = Core.tuple(NoTangent(), %14, %15)
return %16
```

### Putting it together
Now we can assemble everything. The reverse pass is turned into a `Method` by creating an opaque closure from its `CodeInfo` and taking its `source`. We create it with `do_compile = false`, since we never call this closure, we only need its method. The forward pass is then completed: it creates the pullback closure with `Expr(:new_opaque_closure, ...)`, capturing the pullbacks, and returns the tuple `(value, pullback)`. Finally, the forward pass is turned into an opaque closure, which is our generated `rrule`.

```julia
function derive(f, argtypes...; rule = GlobalRef(ChainRules, :rrule))
    ir, R = only(Base.code_ircode(f, argtypes; optimize_until = "CC: COMPACT_1"))
    R = CC.widenconst(R)                  # the type of the output of `f` = the type of Δ
    nargs = length(ir.argtypes)
    fwd, fwd_lines, adinfo, ret = construct_forward(ir; rule)

    # the reverse pass becomes a Method, the body of the pullback closure
    rev, rev_lines = construct_pullback(adinfo, ret, nargs)
    rev_ci = lowered_codeinfo(rev, rev_lines, ir.debuginfo, [Symbol("#self#"), :Δ])
    pullback_method = Core.OpaqueClosure(rev_ci; sig = Tuple{R}, rettype = Any, nargs = 1,
                                         isinferred = false, do_compile = false).source

    # the forward pass creates the pullback closure, capturing the pullbacks
    pullbacks = map(x -> x.pullback_ssa, adinfo)
    push!(fwd, Expr(:new_opaque_closure, Tuple{R}, Union{}, Any, true, pullback_method, pullbacks...))
    append!(fwd_lines, ret.line)
    push!(fwd, Expr(:call, GlobalRef(Core, :tuple), ret.newval, SSAValue(length(fwd))))
    append!(fwd_lines, ret.line)
    push!(fwd, ReturnNode(SSAValue(length(fwd))))
    append!(fwd_lines, ret.line)

    slotnames = [Symbol("#self#"), Base.method_argnames(only(methods(f, argtypes)))[2:end]...]
    fwd_ci = lowered_codeinfo(fwd, fwd_lines, ir.debuginfo, slotnames)
    return Core.OpaqueClosure(fwd_ci; sig = Tuple{argtypes...}, rettype = Any, nargs = nargs - 1,
                              isinferred = false, do_compile = true)
end
```

A few notes:
* `isinferred = false` tells Julia that the code is untyped, so it runs type inference when it compiles the closure.
* The pullback closure takes one argument, `Δ`, which is the gradient with respect to the output of `f`. We declare its type as `Tuple{R}`, where `R` is the type of the output of `f` returned by `code_ircode` (`CC.widenconst` turns a possible `Core.Const` into a type). This matters a lot for performance, see the section on type stability below.
* `nargs` of an opaque closure does not count `#self#`, while `ir.argtypes` does.
* In the forward closure, `Argument(1)` is the closure itself and not the function `f`. This does not matter unless `f` refers to itself (e.g. a recursive call written as `#self#`), which we do not support.

Let's try it

```julia
fwd = derive(foo, Float64, Float64)
y, pullback = fwd(1.0, 2.0)
```

```
julia> y
2.8414709848078967

julia> pullback
(::Float64)->◌::Tuple{ChainRulesCore.NoTangent, Float64, Float64}

julia> pullback(1.0)
(ChainRulesCore.NoTangent(), 2.5403023058681398, 1.0)

julia> pullback.captures
(ChainRules.var"#times_pullback2#rrule##60"{Float64, Float64}(1.0, 2.0), ChainRules.var"#sin_pullback#rrule##47"{Float64}(0.5403023058681398), ChainRules.var"#+_pullback#+_pullback##0"{…}(…))
```

which is correct, since ``\partial foo / \partial x = y + \cos(x) = 2 + \cos(1)`` and ``\partial foo / \partial y = x = 1``. The captured values of the pullback are the pullbacks of `*`, `sin`, and `+`. As always, we define a convenience function `gradient`:

```julia
function gradient(f, args...)
    fwd = derive(f, map(typeof, args)...)
    y, pullback = fwd(args...)
    return y, Base.tail(pullback(one(y)))
end

bar(x, y) = 5 * x
```

```
julia> gradient(foo, 1.0, 2.0)
(2.8414709848078967, (2.5403023058681398, 1.0))

julia> gradient(bar, 1.0, 2.0)
(5.0, (5.0, ChainRulesCore.ZeroTangent()))
```

### Looking at the compiled code
Since Julia's type inference has typed both passes, the compiler could optimize them. Here is the forward pass (`@code_typed debuginfo=:none fwd(1.0, 2.0)`, without the line information):

```
1 ─ %1 = %new(ChainRules.var"#times_pullback2#rrule##60"{Float64, Float64}, x, y)
│   %2 = intrinsic Base.mul_float(x, y)::Float64
│   %3 =    invoke ChainRules.sincos(x::Float64)::Tuple{Float64, Float64}
│   %4 =   builtin Base.getfield(%3, 1)::Float64
│   %5 =   builtin Base.getfield(%3, 2)::Float64
│   %6 = %new(ChainRules.var"#sin_pullback#rrule##47"{Float64}, %5)
│   %7 = intrinsic Base.add_float(%2, %4)::Float64
│   %8 = $(Expr(:new_opaque_closure, Tuple{Float64}, Union{}, Tuple{NoTangent, Float64, Float64}, true, (::Any)(Δ), :(%1), :(%6), …))
│   %9 =   builtin Core.tuple(%7, %8)::Tuple{Float64, Core.OpaqueClosure{Tuple{Float64}}}
└──      return %9
```

All `rrule`s have been inlined: the forward pass multiplies, computes `sincos`, adds, and stores the values needed by the pullbacks. Note that inference has also found the return type of the pullback closure, `Tuple{NoTangent, Float64, Float64}`, which is now the upper bound of its return type. And here is the pullback (`@code_typed pullback(1.0)`):

```
1 ─ %1  =   builtin Core.getfield(#self#, 3)::ChainRules.var"#+_pullback#+_pullback##0"{…}
│   %2  =   builtin Core.getfield(%1, Symbol("#2626#∂f1/∂x1"))::Bool
│   %3  = intrinsic Base.copysign_float(0.0, Δ)::Float64
│   %4  =   builtin (Core.ifelse)(%2, Δ, %3)::Float64
│   %5  =   builtin Core.getfield(%1, Symbol("#2627#∂f1/∂x2"))::Bool
│   %6  = intrinsic Base.copysign_float(0.0, Δ)::Float64
│   %7  =   builtin (Core.ifelse)(%5, Δ, %6)::Float64
│   %8  =   builtin Core.getfield(#self#, 2)::ChainRules.var"#sin_pullback#rrule##47"{Float64}
│   %9  =   builtin Core.getfield(%8, :cosx)::Float64
│   %10 = intrinsic Base.mul_float(%9, %7)::Float64
│   %11 =   builtin Core.getfield(#self#, 1)::ChainRules.var"#times_pullback2#rrule##60"{Float64, Float64}
│   %12 =   builtin Core.getfield(%11, :y)::Float64
│   %13 = intrinsic Base.mul_float(%4, %12)::Float64
│   %14 =   builtin Core.getfield(%11, :x)::Float64
│   %15 = intrinsic Base.mul_float(%14, %4)::Float64
│   %16 = intrinsic Base.add_float(%10, %13)::Float64
│   %17 =   builtin Core.tuple(ChainRulesCore.NoTangent(), %16, %15)::Tuple{ChainRulesCore.NoTangent, Float64, Float64}
└──       return %17
```

When the pullback closure is created, Julia compiles its body for its declared signature `Tuple{Float64}` and for the types of the captured values. Therefore, all three pullbacks from ChainRules have been inlined (`unthunk` of a `Float64` disappears), and the reverse pass is a handful of floating point operations, exactly what we would have written by hand.

Since both passes keep the line information of `foo`, an error thrown inside a pullback can be traced back to the line of `foo` which has caused it. For example, if the pullback of `sin` throws, the stacktrace in Julia 1.13 contains the frame `foo at <file>:<line of z + sin(x)> [inlined]`.

### Opaque closures and type stability
An opaque closure has the type `Core.OpaqueClosure{A, R}`, where `A` is the type of its arguments and `R` its return type. Unlike an ordinary closure, its type does not say *which code* it runs. An ordinary closure is a `struct` with a method, so when the compiler knows its type, it finds its code through dispatch and can inline it. An opaque closure carries its code inside the object, like a function pointer in C. This has consequences for type stability at three places.

**1. Inside the closure.** The body of an opaque closure is compiled *once*, for its declared argument types `A` and the types of the captured values. This is different from an ordinary method, which is compiled for every combination of argument types it is called with. If we had declared the pullback as `Tuple{Any}`, `Δ` would be `Any` in the compiled body, and every call of a pullback from ChainRules would go through dynamic dispatch. Declaring the signature as `Tuple{R}` made the pullback about 10× faster:

| declared signature of the pullback | time of `pullback(1.0)` |
|---|---|
| `Tuple{Any}` | ≈ 110 ns, 288 bytes allocated |
| `Tuple{Float64}` | ≈ 13 ns |

The effect is even larger if the body contains a loop. A closure computing `Δ * z` a thousand times in a loop took ≈ 19 μs with signature `Tuple{Any}` and ≈ 0.4 μs with `Tuple{Float64}`. (If the body immediately calls an ordinary function, this function is specialized by dispatch, and the problem disappears. This is the usual *function barrier*.)

**2. At the place where the closure is called.** If the type of the closure is known, including a concrete `R`, the caller knows the type of the result and the code is type stable. But the call cannot be inlined: it always jumps through the pointer stored in the object.

**3. When the closure leaves the function which has created it.** Inside the forward pass, the compiler knows everything about the pullback closure. But the return type of the forward pass loses the return type of the pullback:

```
julia> Base.return_types(fwd, (Float64, Float64))
1-element Vector{Any}:
 Tuple{Float64, Core.OpaqueClosure{Tuple{Float64}}}
```

although the actual object has type `Core.OpaqueClosure{Tuple{Float64}, Tuple{NoTangent, Float64, Float64}}`. A caller of `fwd` therefore calls the pullback through dynamic dispatch. A function barrier helps again: passing the pullback to a function lets dispatch specialize on its actual type.

Let's measure (with `x = Ref(1.0)`, such that the compiler cannot constant fold the argument):

```julia
using BenchmarkTools, Zygote
grad(fwd, x, y) = ((v, pb) = fwd(x, y); pb(1.0))
call_pullback(pb, Δ) = pb(Δ)
grad_barrier(fwd, x, y) = ((v, pb) = fwd(x, y); call_pullback(pb, 1.0))

@btime $fwd($x[], 2.0)                       # ≈ 30 ns
@btime grad($fwd, $x[], 2.0)                 # ≈ 120–220 ns
@btime grad_barrier($fwd, $x[], 2.0)         # ≈ 65 ns
@btime Zygote.gradient(foo, $x[], 2.0)       # ≈ 6 ns
```

Even with the function barrier, we are about 10× slower than Zygote, although the code of the forward and of the reverse pass is optimal. The remaining cost is the creation of the closure (an allocation) and the two calls through pointers, which cannot be inlined into the caller. Zygote's pullback is an ordinary `struct` whose call is a generated function. The compiler sees through it, and it can inline and optimize everything, including the forward and the reverse pass of the caller. This trade-off is one of the reasons why the performance of Diffractor was uneven. Mooncake also uses opaque closures, but it builds them from typed IR with a precomputed return type, and it gives every rule a concrete type in advance, such that the calls between rules stay type stable.

!!! warning "Do not declare a return type the compiler cannot prove"
    It is tempting to also declare the return type of the forward closure (the `rettype` keyword of `Core.OpaqueClosure`) as the concrete type of `(y, pullback)`. We have tried it, and the forward pass became ≈ 60× slower, because Julia did not compile the closure and ran it in its interpreter (`jl_interpret_opaque_closure` in the profile). Most likely, the declared return type was narrower than what the type inference could prove about the body (see point 3). Declare argument types precisely, and leave the return type as `Any`, unless you know that the inference can prove it.

## Calling functions without rules
So far, every call in the differentiated function had to have an `rrule`. Let's lift this limitation. Notice that our generated function has *exactly the same contract* as `rrule`: it takes the arguments, and it returns the value and a pullback, which returns `(NoTangent(), ∂args...)`. Therefore, the derived rules can be used everywhere where the rules from ChainRules can be used. We define

```julia
const RULES = Dict{Any,Any}()

function ad_rrule(f, args...)
    r = ChainRules.rrule(f, args...)
    r === nothing || return r
    sig = Tuple{typeof(f), map(typeof, args)...}
    fwd = get!(RULES, sig) do
        derive(f, map(typeof, args)...; rule = GlobalRef(Main, :ad_rrule))
    end
    return Base.inferencebarrier(fwd)(args...)
end

function ad_gradient(f, args...)
    y, pullback = ad_rrule(f, args...)
    return y, map(ChainRules.unthunk, Base.tail(pullback(one(y))))
end
```

`ad_rrule` uses the `rrule` from ChainRules if there is one (ChainRules returns `nothing` if there is not). Otherwise, it derives a rule, in which every call is replaced by a call to `ad_rrule`. Hence, the functions called from `f` are differentiated in the same way, recursively. The derived rules are cached in `RULES`, such that every rule is derived only once. Note that a rule is derived *lazily*, only when it is called for the first time. If we derived the rules of all called functions when deriving the rule of `f`, a recursive function would lead to an infinite loop.

```julia
g(x) = foo(x, x) * x
h(x) = g(sin(x)) + x
k(x, y) = h(x) * foo(y, x)
```

```
julia> ad_gradient(g, 2.0)
(9.818594853651364, (12.077003753731397,))

julia> ad_gradient(k, 0.3, 1.2)
(0.5321586079069127, (2.8181212931991033, 0.27280860493069964))

julia> keys(RULES)
KeySet for a Dict{Any, Any} with 4 entries. Keys:
  Tuple{typeof(k), Float64, Float64}
  Tuple{typeof(h), Float64}
  Tuple{typeof(g), Float64}
  Tuple{typeof(foo), Float64, Float64}
```

The results agree with Zygote.

!!! warning "`Base.inferencebarrier`"
    `Base.inferencebarrier(fwd)` hides the type of `fwd` from the compiler, so the closure is called through dynamic dispatch. Since `RULES` holds values of type `Any`, the call is dynamic anyway. But there is one more reason. We have observed segmentation faults (in Julia 1.13 and 1.14) when an opaque closure created from generated code was called directly from compiled code with certain argument types. Mooncake also calls its rules through `inferencebarrier`.

The price of this flexibility is performance. Since `RULES` is a `Dict{Any,Any}`, the compiler does not know which rule is called, and the types of the pullbacks captured by the closures are not known. For `k`, the gradient takes ≈ 6–7 μs, while Zygote takes ≈ 20 ns.

There is also a catch in combining derived rules with ChainRules. A derived pullback accepts only gradients of type `R`, the type of the output of its function. For a function returning `Float64`, it accepts a `Float64`, but not, e.g., a `ZeroTangent()`, which ChainRules uses for a gradient which is known to be zero. We handle thunks by `unthunk`, but a robust system has to fix the type of the gradient as a function of the type of the value. This is what Mooncake does: the type of the tangent of a value of type `T` is given by `tangent_type(T)`. Mooncake solves this by computing the type of every rule in advance, and by wrapping a not-yet-derived rule in a typed object (`LazyDerivedRule`), which derives the rule on its first call.

## What is missing
Our AD system does not support control flow. The extension to functions with branches and loops (and to the recursion, which needs control flow to terminate) is shown in [full_ad.md](full_ad.md). That version records the pullbacks on a tape and the executed basic blocks on a trace. With the approach of this lecture, the pullback closure would capture the tape and the trace instead of the individual pullbacks. Diffractor did something more elaborate: it put the pullbacks of every basic block into a tuple, and linked these tuples into a list through phi nodes, such that the list records the executed path.

## References
* [Autodiff by G. Dalle](https://gdalle.github.io/JuliaOptimizationDays2024-AutoDiff/#/)
* [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl), in particular `src/stage1/generated.jl` and `src/codegen/reverse.jl`
* [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl)
