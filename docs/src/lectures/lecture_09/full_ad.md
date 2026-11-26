# A (Nearly) Full Source-to-Source AD

In the [lecture](lecture_v4.md) we have built a source-to-source AD which rewrites the code of a function, and which creates the pullback as an opaque closure. It was simple, but it had two big limitations:

1. **Limited recursion.** At the end of the lecture, we have shown how to differentiate functions calling functions without `rrule`, but only if all of them consist of a single basic block. Since a recursive function needs a branch to terminate, recursive functions were not supported.
2. **No control flow.** The function had to consist of a single basic block, i.e. no `if`, no loops.

In this text we remove both. The result is an AD which can differentiate functions calling other functions, branches, `while` and `for` loops, recursive functions, and functions with tuples and varargs. The code has about 250 lines and its design follows [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl), although it is much simpler (and much slower).

!!! info "Tested versions"
    The code was tested with Julia 1.13.4 and Julia 1.14.0-beta1, ChainRules 1.74 and ChainRulesCore 1.26. As in the lecture, it relies on compiler internals, which are not part of the public API.

## The design

Recall that an `rrule` for a function `f` returns the value `f(args...)` and a *pullback*. We will write a function

```
ad_rrule(f, args...)
```

with the same contract, which works for any `f`:

* If ChainRules has an `rrule` for `f`, we use it.
* If it does not, we *derive* a rule from the `IRCode` of `f`. Every call inside `f` is replaced by a call to `ad_rrule`. This is how **recursion** works: the derived code calls `ad_rrule` again, which either finds an `rrule`, or derives another rule.

Notice that deriving a rule for `f` does *not* derive rules for the functions `f` calls. These rules are derived *lazily*, when the forward pass of `f` actually executes the call. This has two advantages. First, we never derive rules for code which is never executed. Second, recursive functions work. If we derived the rules eagerly, a recursive function `fact(x, n) = n == 0 ? one(x) : x * fact(x, n - 1)` would lead to an infinite loop, as deriving the rule of `fact` would require deriving the rule of `fact`. Mooncake does the same with its `LazyDerivedRule`.

To support **control flow**, the forward pass keeps the structure of the original function (its blocks, branches and loops) and records two things:

* a **tape**: the pullbacks of all executed calls, in the order of execution;
* a **trace**: the sequence of executed basic blocks.

As in the lecture, the forward pass returns the value and the **pullback closure**. This time, the closure captures the tape and the trace. Its body, the reverse pass, walks the trace backwards. For every block it pops the pullbacks of that block from the tape (again backwards), and accumulates the gradients.

Since loops can execute the same statement many times, we cannot store gradients in SSA values of the reverse pass. Instead, the gradients are stored in a vector `grads` with one slot for every SSA value and every argument of the original function.

As in the lecture, both passes are handed to Julia as *untyped* code, which is typed by Julia's type inference when the closures are compiled. Therefore, we do not need the type inference on `IRCode` (`infer_ir!`) at all.

```julia
import Core.Compiler as CC
using ChainRules, ChainRulesCore
using Core: SSAValue, Argument, GlobalRef, ReturnNode, GotoNode, GotoIfNot, PhiNode
```

## A running example

We will use the following function, which has a branch:

```julia
function f(x)
    if x > 0
        y = sin(x)
    else
        y = x * x
    end
    return y * x
end
```

Its `IRCode` (obtained by `Base.code_ircode_by_type(Tuple{typeof(f), Float64}; optimize_until = "CC: COMPACT_1")`) has four basic blocks:

```
3 1 ─ %1 =   dynamic (_2 > 0)::Bool
  └──      goto #3 if not %1
4 2 ─ %3 =   dynamic Main.sin(_2)::Float64
  └──      goto #4
6 3 ─ %5 =   dynamic (_2 * _2)::Float64
8 4 ┄ %6 = φ (#2 => %3, #3 => %5)::Float64
  │   %7 =   dynamic (%6 * _2)::Float64
  └──      return %7
```

Block 1 evaluates the condition and either falls through to block 2 (`sin`) or jumps to block 3 (`x * x`). Both branches meet in block 4, where the *phi node* `%6` selects the value of `y` according to the block we came from: `%3` if we came from block 2, `%5` if we came from block 3.

## Which calls do we differentiate?

Not every call needs a pullback. `x > 0` returns a `Bool`, which carries no gradient, and loops are full of calls like `iterate(1:n, i)` working with integers. We call a statement **active** if it is a call, its result can carry a gradient, and at least one of its arguments can carry a gradient. This is a (very) simplified version of the *activity analysis* used by Enzyme and Mooncake. We decide it from the types inferred by the compiler, so the decision is made once, when the rule is derived.

```julia
const NonDiff = Union{Integer, Nothing, Symbol, AbstractString, Type, Function,
                      AbstractRange{<:Integer}, Tuple{Vararg{Integer}}}
isnondiff(@nospecialize(T)) = CC.widenconst(T) <: NonDiff

function isactive(ir, i)
    stmt = ir.stmts.stmt[i]
    Meta.isexpr(stmt, :call) || return false
    isnondiff(ir.stmts.type[i]) && return false
    return any(a -> !isnondiff(CC.argextype(a, ir)), stmt.args[2:end])
end
```

`CC.argextype(a, ir)` returns the type of an argument of a statement, whatever it is (an `SSAValue`, an `Argument`, a `GlobalRef` or a constant). In our example, statements `%3`, `%5`, and `%7` are active, while `%1` (`x > 0`) is not.

Note that `Function` is in `NonDiff`. This means that we never compute gradients with respect to the function itself. For closures, gradients with respect to the captured variables are therefore lost.

## Runtime helpers

The generated code calls a few small helper functions. Keeping them as ordinary Julia functions keeps the generated code short and readable.

```julia
new_tape() = Any[]
new_trace() = Int[]
record!(tape, r) = (push!(tape, r[2]); r[1])   # store the pullback, return the value
pop_block!(trace) = isempty(trace) ? 0 : pop!(trace)
last_block(trace) = isempty(trace) ? 0 : last(trace)
accum(a, b) = a + unthunk(b)
zero_grads(n) = Any[ZeroTangent() for _ in 1:n]
function take_grad!(grads, i)
    g = grads[i]
    grads[i] = ZeroTangent()
    return g
end
accum_grad!(grads, i, Δ) = (grads[i] = accum(grads[i], Δ); nothing)
```

`ZeroTangent()` from ChainRulesCore is a "strong zero": `ZeroTangent() + x == x` for any `x`, so we do not need to know the type of a gradient in advance.

## The reverse pass

We start with the reverse pass, since the forward pass needs it: the forward pass creates the pullback closure, whose body is the reverse pass. As in the lecture, the reverse pass is generated as untyped code (`CodeInfo`) with two arguments: `Argument(1)` is the closure itself, whose captured values are the tape (`getfield(Argument(1), 1)`) and the trace (`getfield(Argument(1), 2)`), and `Argument(2)` is the gradient `Δ` of the output. Its structure is:

* An **entry block** reading the tape and the trace, and creating the vector of gradients `grads`. The slot of SSA value `%i` is `i`, and the slot of `Argument(n)` is `nssa + n`.
* A **dispatch block**, which pops the last block number `b` from the trace and jumps to the reverse of block `b`. If the trace is empty, it jumps to the exit block.
* For every block `b` of the original code, a **reverse block**, which:
  1. if the block ends with `return v`, adds `Δ` to the gradient of `v`;
  2. goes through the active statements of the block backwards. For each one, it pops the pullback from the tape, takes the gradient of the statement's result (`take_grad!` also resets it to zero), calls the pullback, and adds the gradients of the arguments to their slots;
  3. handles the phi nodes (see below);
  4. jumps back to the dispatch block.
* An **exit block** returning the gradients of the arguments, including the gradient with respect to `f` itself, as required by the ChainRules convention.

Why does `take_grad!` reset the gradient to zero? In a loop, the same statement `%i` is executed many times, and each execution produces a *different* value. Since SSA values are defined before they are used, all uses of the value produced by the `k`-th execution happen before its `(k+1)`-th execution. Therefore, when the reverse pass reaches the statement, `grads[i]` contains exactly the gradient of the value produced by the corresponding execution, and the slot has to be cleared for the previous execution.

**Phi nodes** are the reverse of the branches. The phi node `%6 = φ (#2 => %3, #3 => %5)` passes its value to `%3` if we came to block 4 from block 2, and to `%5` if we came from block 3. Since we have just popped block 4 from the trace, the block we came from is the *last* element of the remaining trace. All phi nodes at the beginning of a block are evaluated *at the same time*, so we first take the gradients of all of them and only then pass them to their predecessors. (If we processed them one by one, a phi node referring to another phi node of the same block, as in `x, y = y, x` in a loop, would receive the wrong gradient.)

**Varargs.** A method like `g(x, ys...)` has, in its `IRCode`, an argument `ys` which is a tuple. The pullback has to return the gradients with respect to the individual arguments `y1, y2, ...`, so the reverse pass unpacks the gradient with respect to the tuple at the end (`nva` is the number of the variable arguments).

In lowered code, the targets of jumps (`GotoNode`, `GotoIfNot`) are *statement indices* (in `IRCode` they are block numbers). Since we emit statements one by one, statement indices are what we know anyway, and we do not need any conversion. The helper `lowered_codeinfo` is the same as in the lecture, except that the reverse pass has no line information:

```julia
function lowered_codeinfo(insts::Vector{Any}, slotnames::Vector{Symbol})
    n = length(insts)
    ci = ccall(:jl_new_code_info_uninit, Ref{Core.CodeInfo}, ())
    ci.code = insts
    ci.ssavaluetypes = n                  # the code is not inferred
    ci.ssaflags = zeros(UInt32, n)
    ci.slotnames = slotnames
    ci.slotflags = zeros(UInt8, length(slotnames))
    ci.nargs = length(slotnames)
    ci.isva = false
    di = CC.DebugInfoStream(zeros(Int32, 3n))   # no line information
    di.def = :pullback
    ci.debuginfo = Core.DebugInfo(di, n)
    return ci
end

function unpack_vararg(∂, nva)
    ∂va = last(∂)
    ∂va = ∂va isa AbstractZero ? ntuple(_ -> ∂va, nva) : ChainRulesCore.backing(∂va)
    return (Base.front(∂)..., ∂va...)
end

function reverse_ir(primal::CC.IRCode, active, isva, nva)
    nssa, nargs = length(primal.stmts), length(primal.argtypes)
    slot(x::SSAValue) = x.id
    slot(x::Argument) = nssa + x.n
    isvar(x) = x isa Union{SSAValue, Argument}
    call(f, args...) = Expr(:call, f, args...)
    main(f) = GlobalRef(Main, f)
    insts = Any[]
    emit!(x) = (push!(insts, x); SSAValue(length(insts)))

    Δ = Argument(2)
    tape = emit!(call(GlobalRef(Core, :getfield), Argument(1), 1))     # captured values
    trace = emit!(call(GlobalRef(Core, :getfield), Argument(1), 2))
    grads = emit!(call(main(:zero_grads), nssa + nargs))

    # dispatch: pop the block executed last and jump to its reverse
    blocks = primal.cfg.blocks
    dispatch = length(insts) + 1
    b = emit!(call(main(:pop_block!), trace))
    jumps = Pair{Int,Int}[]
    for k in eachindex(blocks)
        c = emit!(call(GlobalRef(Core, :(===)), b, k))
        emit!(GotoIfNot(c, length(insts) + 3))
        push!(jumps, emit!(nothing).id => k)
    end
    push!(jumps, emit!(nothing).id => 0)       # trace is empty => exit

    starts = Dict{Int,Int}()
    for k in eachindex(blocks)
        starts[k] = length(insts) + 1
        rng = blocks[k].stmts
        term = primal.stmts.stmt[last(rng)]
        if term isa ReturnNode && isdefined(term, :val) && isvar(term.val)
            emit!(call(main(:accum_grad!), grads, slot(term.val), Δ))
        end
        for i in reverse(rng)
            stmt = primal.stmts.stmt[i]
            if active[i]
                pb = emit!(call(GlobalRef(Base, :pop!), tape))
                ȳ = emit!(call(main(:take_grad!), grads, i))
                ∂ = emit!(call(pb, ȳ))
                for (j, a) in enumerate(stmt.args)
                    (j == 1 || !isvar(a)) && continue
                    ∂a = emit!(call(GlobalRef(Core, :getfield), ∂, j))
                    emit!(call(main(:accum_grad!), grads, slot(a), ∂a))
                end
            end
        end
        # phi nodes of a block are evaluated in parallel: first take all their gradients,
        # then send them to the values coming from the predecessor block
        phis = filter(i -> primal.stmts.stmt[i] isa PhiNode, rng)
        if !isempty(phis)
            pred = emit!(call(main(:last_block), trace))
            ȳs = [emit!(call(main(:take_grad!), grads, i)) for i in phis]
            for (i, ȳ) in zip(phis, ȳs)
                phi = primal.stmts.stmt[i]
                for (e, v) in zip(phi.edges, phi.values)
                    isvar(v) || continue
                    c = emit!(call(GlobalRef(Core, :(===)), pred, Int(e)))
                    ∂v = emit!(call(GlobalRef(Core, :ifelse), c, ȳ, ZeroTangent()))
                    emit!(call(main(:accum_grad!), grads, slot(v), ∂v))
                end
            end
        end
        emit!(GotoNode(dispatch))
    end

    starts[0] = length(insts) + 1
    ∇args = [emit!(call(GlobalRef(Base, :getindex), grads, nssa + n)) for n in 1:nargs]
    ∇ = emit!(call(GlobalRef(Core, :tuple), ∇args...))
    if isva   # the last gradient belongs to the tuple of varargs, unpack it
        ∇ = emit!(call(main(:unpack_vararg), ∇, nva))
    end
    emit!(ReturnNode(∇))

    for (i, k) in jumps
        insts[i] = GotoNode(starts[k])     # jump targets are statement indices in lowered code
    end
    return lowered_codeinfo(insts, [Symbol("#self#"), :Δ])
end
```

For our function `f`, the reverse pass looks like this. The original function has 8 statements (including `return`) and 2 arguments (`f` itself and `x`), so `grads` has 10 slots: slots 1–8 belong to `%1`–`%8`, slot 9 to `f`, and slot 10 to `x`. (The printed block numbers are computed by the printer; in the code, jumps target statement indices.)

```
1 ── %1  =   builtin Core.getfield(#self#, 1)                    # tape
│    %2  =   builtin Core.getfield(#self#, 2)                    # trace
└─── %3  =   dynamic Main.zero_grads(10)                         # grads
2 ┄─ %4  =   dynamic Main.pop_block!(%2)                         # dispatch
│    %5  =   builtin %4 === 1
└───       goto #4 if not %5
3 ──       goto #11
4 ── %8  =   builtin %4 === 2
└───       goto #6 if not %8
5 ──       goto #12
6 ── %11 =   builtin %4 === 3
└───       goto #8 if not %11
7 ──       goto #13
8 ── %14 =   builtin %4 === 4
└───       goto #10 if not %14
9 ──       goto #14
10 ─       goto #15
11 ─       goto #2                                               # reverse of block 1 (nothing to do)
12 ─ %19 =   dynamic Base.pop!(%1)                               # reverse of block 2: sin
│    %20 =   dynamic Main.take_grad!(%3, 3)
│    %21 =   dynamic (%19)(%20)
│    %22 =   builtin Core.getfield(%21, 2)
│            dynamic Main.accum_grad!(%3, 10, %22)
└───       goto #2
13 ─ %25 =   dynamic Base.pop!(%1)                               # reverse of block 3: x * x
│    %26 =   dynamic Main.take_grad!(%3, 5)
│    %27 =   dynamic (%25)(%26)
│    %28 =   builtin Core.getfield(%27, 2)
│            dynamic Main.accum_grad!(%3, 10, %28)
│    %30 =   builtin Core.getfield(%27, 3)
│            dynamic Main.accum_grad!(%3, 10, %30)
└───       goto #2
14 ─         dynamic Main.accum_grad!(%3, 7, Δ)                  # reverse of block 4: return
│    %34 =   dynamic Base.pop!(%1)                               #   y * x
│    %35 =   dynamic Main.take_grad!(%3, 7)
│    %36 =   dynamic (%34)(%35)
│    %37 =   builtin Core.getfield(%36, 2)
│            dynamic Main.accum_grad!(%3, 6, %37)
│    %39 =   builtin Core.getfield(%36, 3)
│            dynamic Main.accum_grad!(%3, 10, %39)
│    %41 =   dynamic Main.last_block(%2)                         #   φ
│    %42 =   dynamic Main.take_grad!(%3, 6)
│    %43 =   builtin %41 === 2
│    %44 =   builtin Core.ifelse(%43, %42, ZeroTangent())
│            dynamic Main.accum_grad!(%3, 3, %44)
│    %46 =   builtin %41 === 3
│    %47 =   builtin Core.ifelse(%46, %42, ZeroTangent())
│            dynamic Main.accum_grad!(%3, 5, %47)
└───       goto #2
15 ─ %50 =   dynamic Base.getindex(%3, 9)                        # exit: ∂f, ∂x
│    %51 =   dynamic Base.getindex(%3, 10)
│    %52 =   builtin Core.tuple(%50, %51)
└───       return %52
```

For `x = 2.0`, the trace is `[1, 2, 4]`, so the reverse pass executes blocks 14 (reverse of 4), 12 (reverse of 2), 11 (reverse of 1), and exits. Block 13 (the reverse of the branch not taken) is never executed.

## The forward pass

The forward pass is a *modified copy* of the original IR. Unlike in the lecture, we do not construct the statements from scratch. We use the compiler's API for editing `IRCode`:

* `CC.insert_node!(ir, i, CC.NewInstruction(stmt, type))` inserts a new statement before statement `i` and returns an `SSAValue` referring to it. The new statements are kept in `ir.new_nodes`, so the existing statements keep their numbers and we can freely refer to them.
* `CC.compact!(ir)` puts the new statements in their correct places and renumbers all statements, including the targets of jumps and the edges of phi nodes.

This is the infrastructure the compiler itself uses for inlining, and it is the main reason why keeping the control flow is easy: we only add statements *inside* blocks and never change the blocks themselves.

The forward pass does the following:

1. At the beginning, it creates an empty `tape` and an empty `trace`.
2. At the beginning of every block (after the phi nodes), it records the block number: `push!(trace, b)`.
3. Every active call `%i = f(args...)` is replaced by
   ```
   r  = ad_rrule(f, args...)
   y  = record!(tape, r)
   %i = Core.typeassert(y, T)
   ```
   where `T` is the type of `%i` inferred in the original code. Since `%i` still holds the value of `f(args...)`, the other statements, including phi nodes, do not need to change. The `typeassert` tells the compiler that the value has the same type as in the original code. Without it, the value would have the type returned by `ad_rrule`, which is `Any` for derived rules, and the rest of the forward pass would not be type stable.
4. Every `return v` is replaced by
   ```
   pb = $(Expr(:new_opaque_closure, Tuple{Any}, Union{}, Any, true, pullback_method, tape, trace))
   return (v, pb)
   ```
   i.e. it creates the pullback closure capturing the tape and the trace, exactly as in the lecture.

There is one more detail. In the lecture, `Argument(1)` of the forward closure was the closure itself, and not the function `f`. But some methods do refer to themselves. For example, `iterate(A::Array) = iterate(A, 1)` calls `#self#`. We therefore pass `f` to the opaque closure as an ordinary argument: all arguments are shifted by one, `Argument(1)` is the closure itself, and `Argument(2)` is `f`.

```julia
function first_nonphi(ir, b)
    for i in ir.cfg.blocks[b].stmts
        ir.stmts.stmt[i] isa PhiNode || return i
    end
end

shift_args(x::Argument) = Argument(x.n + 1)
shift_args(x::Expr) = Expr(x.head, map(shift_args, x.args)...)
shift_args(x::ReturnNode) = isdefined(x, :val) ? ReturnNode(shift_args(x.val)) : x
shift_args(x::GotoIfNot) = GotoIfNot(shift_args(x.cond), x.dest)
shift_args(x::PhiNode) = PhiNode(x.edges, Any[isassigned(x.values, i) ? shift_args(x.values[i]) : nothing for i in eachindex(x.values)])
shift_args(x) = x

function forward_ir(primal::CC.IRCode, active, pullback_method)
    ir = CC.copy(primal)
    # the function `f` becomes an ordinary argument of the opaque closure
    map!(shift_args, ir.stmts.stmt, ir.stmts.stmt)
    pushfirst!(ir.argtypes, Tuple{})
    new(stmt) = CC.NewInstruction(stmt, Any)
    p1 = first_nonphi(ir, 1)
    tape = CC.insert_node!(ir, p1, new(Expr(:call, GlobalRef(Main, :new_tape))))
    trace = CC.insert_node!(ir, p1, new(Expr(:call, GlobalRef(Main, :new_trace))))
    for b in eachindex(ir.cfg.blocks)
        CC.insert_node!(ir, first_nonphi(ir, b), new(Expr(:call, GlobalRef(Base, :push!), trace, b)))
    end
    for i in 1:length(ir.stmts)
        stmt = ir.stmts.stmt[i]
        if active[i]
            r = CC.insert_node!(ir, i, new(Expr(:call, GlobalRef(Main, :ad_rrule), stmt.args...)))
            y = CC.insert_node!(ir, i, new(Expr(:call, GlobalRef(Main, :record!), tape, r)))
            # the value has the same type as in the original code; tell it to the compiler
            ir.stmts.stmt[i] = Expr(:call, GlobalRef(Core, :typeassert), y, CC.widenconst(ir.stmts.type[i]))
        elseif stmt isa ReturnNode && isdefined(stmt, :val)
            pb = CC.insert_node!(ir, i, new(Expr(:new_opaque_closure, Tuple{Any}, Union{}, Any, true, pullback_method, tape, trace)))
            ret = CC.insert_node!(ir, i, new(Expr(:call, GlobalRef(Core, :tuple), stmt.val, pb)))
            ir.stmts.stmt[i] = ReturnNode(ret)
        end
    end
    return CC.compact!(ir)
end
```

For our function `f`, the forward pass looks like this:

```
4 1 ─ %1  =   dynamic Main.new_tape()::Any
  │   %2  =   dynamic Main.new_trace()::Any
  │           dynamic Base.push!(%2, 1)::Any
  │   %4  =   dynamic (_3 > 0)::Bool
  └──       goto #3 if not %4
5 2 ─         dynamic Base.push!(%2, 2)::Any
  │   %7  =   dynamic Main.ad_rrule(Main.sin, _3)::Any
  │   %8  =   dynamic Main.record!(%1, %7)::Any
  │   %9  =   builtin Core.typeassert(%8, Float64)::Float64
  └──       goto #4
7 3 ─         dynamic Base.push!(%2, 3)::Any
  │   %12 =   dynamic Main.ad_rrule(Main.:*, _3, _3)::Any
  │   %13 =   dynamic Main.record!(%1, %12)::Any
  └── %14 =   builtin Core.typeassert(%13, Float64)::Float64
9 4 ┄ %15 = φ (#2 => %9, #3 => %14)::Float64
  │           dynamic Base.push!(%2, 4)::Any
  │   %17 =   dynamic Main.ad_rrule(Main.:*, %15, _3)::Any
  │   %18 =   dynamic Main.record!(%1, %17)::Any
  │   %19 =   builtin Core.typeassert(%18, Float64)::Float64
  │   %20 = $(Expr(:new_opaque_closure, Tuple{Any}, Union{}, Any, true, (::Any)(Δ) @ Base.Experimental :0, :(%1), :(%2)))::Any
  │   %21 =   builtin Core.tuple(%19, %20)::Any
  └──       return %21
```

Observe that:

* the blocks, the branch, and the phi node are the same as in the original code; only the statement numbers have changed;
* `x` is now `_3`, since `_2` is the function `f`;
* the statements keep their line numbers, since `insert_node!` copies the debuginfo of the statement at the insertion point (see [debuginfo.md](debuginfo.md));
* the types of the new statements are `Any`. We do not run the type inference on `IRCode`, since the IR interpreter does not understand `:new_opaque_closure` (see the lecture). Julia types the code when the closure is compiled.

### From `IRCode` to untyped `CodeInfo`

To hand the forward pass to Julia as untyped code, we convert the `IRCode` to `CodeInfo` with the compiler's function `CC.ir_to_codeinf!`, which is the same function `Core.OpaqueClosure(ir::IRCode)` uses. It converts the block numbers of jumps to statement indices and compresses the debuginfo. Then we throw away the types of the statements (`ssavaluetypes = length(code)`), which tells Julia that the code has not been inferred.

```julia
function ircode_to_lowered(ir::CC.IRCode, isva::Bool)
    ci = ccall(:jl_new_code_info_uninit, Ref{Core.CodeInfo}, ())
    n = length(ir.argtypes)
    ci.slotnames = [Symbol("#self#"), (Symbol("arg", i) for i in 2:n)...]
    ci.slotflags = zeros(UInt8, n)
    ci.slottypes = copy(ir.argtypes)
    ci.nargs = n
    ci.isva = isva
    CC.ir_to_codeinf!(ci, ir)                 # code, jump targets as statement indices, debuginfo
    ci.ssavaluetypes = length(ci.code)        # forget the types: the code is not inferred
    ci.slottypes = nothing
    return ci
end
```

## Rules

A derived rule is now simply the forward closure: calling it with `(f, args...)` returns the value and the pullback closure, exactly like `rrule`. Rules are derived once per signature and cached.

```julia
const RULES = Dict{Any,Any}()

# Builtins (getfield, setfield!, memoryrefset!, ...) have no Julia code, so an active call
# to a builtin needs an rrule. We check this when the rule is derived, not when it runs.
function check_builtins(ir, active)
    for i in findall(active)
        args = ir.stmts.stmt[i].args
        f = CC.argextype(args[1], ir)
        f isa Core.Const && f.val isa Core.Builtin || continue
        sig = Tuple{typeof(f.val), map(a -> CC.widenconst(CC.argextype(a, ir)), args[2:end])...}
        m = which(ChainRules.rrule, sig)
        if m.sig == Tuple{typeof(ChainRules.rrule), Any, Vararg{Any}}
            error("Cannot differentiate the builtin call $(ir.stmts.stmt[i]) with argument types $(sig): it has no rrule.")
        end
    end
end

function derive_rule(sig)
    get!(RULES, sig) do
        irs = Base.code_ircode_by_type(sig; optimize_until = "CC: COMPACT_1")
        if length(irs) != 1 || !(first(first(irs)) isa CC.IRCode)
            error("There is no rrule for $sig and there is no Julia code to differentiate.")
        end
        primal, _ = only(irs)
        isva = Base._which(sig).method.isva
        nva = length(sig.parameters) - (length(primal.argtypes) - isva)
        active = [isactive(primal, i) for i in 1:length(primal.stmts)]
        check_builtins(primal, active)

        # the reverse pass becomes a Method, the body of the pullback closure
        rvs = reverse_ir(primal, active, isva, nva)
        pullback_method = Core.OpaqueClosure(rvs; sig = Tuple{Any}, rettype = Any, nargs = 1,
                                             isinferred = false, do_compile = false).source

        # the forward pass creates the pullback closure and is our rule
        fwd = forward_ir(primal, active, pullback_method)
        argtypes = map(CC.widenconst, fwd.argtypes[2:end])
        isva && (argtypes = [argtypes[1:end-1]..., last(argtypes).parameters...])   # unpack the varargs
        Core.OpaqueClosure(ircode_to_lowered(fwd, isva); sig = Tuple{argtypes...}, rettype = Any,
                           nargs = length(fwd.argtypes) - 1, isva, isinferred = false, do_compile = true)
    end
end
```

A few details deserve a comment:

* **`check_builtins`.** *Builtins* such as `getfield`, `tuple`, or `Core.memoryrefset!` are implemented in C and have no Julia code, so there is nothing to derive a rule from. When the derived code calls a builtin with active arguments and there is no `rrule` for it, we cannot differentiate it. We could find out when `ad_rrule` is called at runtime, but checking when the rule is derived gives better error messages. (We also hit segmentation faults in Julia 1.14 when the AD went deep into array internals before failing.) `which(ChainRules.rrule, sig)` tells us which method of `rrule` would be called. If it is the generic fallback `rrule(::Any, ::Vararg{Any})`, which returns `nothing`, there is no rule.
* **Varargs.** For a vararg method like `g(x, ys...)`, the forward closure is created with `isva = true`, and its signature lists the variable arguments individually, such that it can be called as `fwd(g, x, y1, y2)`.
* **The signature of the pullback is `Tuple{Any}`.** In the lecture, we have declared the type of `Δ` precisely, since the body of an opaque closure is compiled only for its declared signature (see the section on type stability in the lecture). Here, `Tuple{Any}` is deliberate. The gradients live in the untyped vector `grads` anyway, and the pullback of a derived rule may receive a `ZeroTangent()` (e.g. from `take_grad!` of a value whose gradient has not been set), which a typed signature would reject.

Finally, `ad_rrule` puts everything together:

```julia
function ad_rrule(f, args...)
    r = ChainRules.rrule(f, args...)
    r === nothing || return r
    fwd = derive_rule(Tuple{Core.Typeof(f), map(Core.Typeof, args)...})
    return Base.inferencebarrier(fwd)(f, args...)
end
```

`Core.Typeof(x)` is `typeof(x)`, except for types, where it returns `Type{x}`. This matches how Julia dispatches on types. We call the forward closure through `Base.inferencebarrier`, which hides its type from the compiler, so that it is called by dynamic dispatch. We have observed segmentation faults (in Julia 1.13 and 1.14) when an opaque closure created from generated code was called directly from compiled code with certain argument types (e.g. `(Vector{Float64}, Symbol)`). Mooncake also uses `inferencebarrier` when calling its rules. Since `RULES` holds values of type `Any`, the call would be dynamic anyway.

### Rules for builtins

ChainRules does not have rules for the builtins `getfield` and `tuple`, which are needed for tuples, including destructuring like `a, b = sin(x), cos(x)` and iteration over vectors (`iterate` returns a tuple `(value, state)`). We define them. The gradient with respect to a tuple is a `Tangent{<:Tuple}`, and `ChainRulesCore.backing` extracts the tuple of gradients from it.

```julia
function ChainRules.rrule(::typeof(getfield), t::Tuple, i::Int)
    getfield_pullback(Δ) = (NoTangent(), Tangent{typeof(t)}(ntuple(j -> j == i ? Δ : ZeroTangent(), length(t))...), NoTangent())
    return getfield(t, i), getfield_pullback
end

function ChainRules.rrule(::typeof(tuple), args...)
    tuple_pullback(Δ::Tangent) = (NoTangent(), ChainRulesCore.backing(Δ)...)
    tuple_pullback(Δ::AbstractZero) = (NoTangent(), map(_ -> Δ, args)...)
    return args, tuple_pullback
end
```

!!! warning "Type piracy"
    Defining `rrule(::typeof(getfield), ...)` is type piracy: we define a method of a function we do not own for types we do not own. It is fine for a lecture, but a real AD system would use its own rule function (Mooncake has `rrule!!`, Zygote has `@adjoint`) and fall back to ChainRules.

## Using it

```julia
function gradient(f, args...)
    y, pb = ad_rrule(f, args...)
    return y, map(unthunk, Base.tail(pb(one(y))))
end
```

`Base.tail` drops the gradient with respect to `f` itself. Let's try our example:

```julia
gradient(f, 2.0)     # (1.8185948536513634, (0.0770037537313969,))  ==  sin(2) + 2cos(2)
gradient(f, -2.0)    # (-8.0, (12.0,))                              ==  3x^2
```

The pullback is the opaque closure created by the forward pass. Its captured values are the tape and the trace:

```julia
y, pb = ad_rrule(f, 2.0)
pb.captures[2]    # [1, 2, 4]: the trace of executed blocks
```

Since Julia's type inference types the forward pass as a whole, it also sees through `ad_rrule` when ChainRules has a rule. If you look at `@code_typed derive_rule(Tuple{typeof(f), Float64})(f, 2.0)`, you will find that the calls of `rrule` for `sin` and `*` have been inlined into `sincos` and `mul_float`, and what remains are the operations on the tape and the trace.

Functions calling functions:

```julia
function foo(x, y)
    z = x * y
    z + sin(x)
end
g(x) = foo(x, x) * x
gradient(g, 2.0)     # (9.818594853651364, (12.077003753731397,))
```

Loops. Calls on integers (`1:n`, `iterate`, `===`) are not active, so the only pullbacks on the tape are those of `r *= x`:

```julia
function mypow(x, n)
    r = one(x)
    for i in 1:n
        r *= x
    end
    return r
end
gradient(mypow, 2.0, 5)    # (32.0, (80.0, ZeroTangent()))
```

Recursion and mutual recursion:

```julia
fact(x, n) = n == 0 ? one(x) : x * fact(x, n - 1)
gradient(fact, 2.0, 5)     # (32.0, (80.0, ZeroTangent()))

iseven_(x, n) = n == 0 ? x : isodd_(x * x, n - 1)
isodd_(x, n) = n == 0 ? sin(x) : iseven_(x + 1, n - 1)
gradient(iseven_, 0.3, 5)  # (-0.9971593216284981, (0.4311439477043812, ZeroTangent()))
```

`while` loops with nested branches:

```julia
function collatz_like(x, n)
    while n > 1
        if iseven(n)
            x = sin(x) * x
            n = n ÷ 2
        else
            x = x + cos(x)
            n = 3n + 1
        end
    end
    x
end
gradient(collatz_like, 0.7, 6)   # (1.5681050757655928, (0.027224862170209356, ZeroTangent()))
```

Iterating over a vector. Here `iterate(xs, i)` returns `(xs[i], i + 1)`, so the rules for `tuple` and `getfield` are needed, and `getindex` uses the rule from ChainRules:

```julia
function mysum(xs)
    s = zero(eltype(xs))
    for x in xs
        s += sin(x)
    end
    s
end
gradient(mysum, [1.0, 2.0, 3.0])  # (1.8918884196934453, ([0.5403023058681398, -0.4161468365471424, -0.9899924966004454],))
```

Varargs:

```julia
va(x, ys...) = x * ys[1] + ys[2]
gradient(va, 1.0, 2.0, 3.0)      # (5.0, (2.0, 1.0, 1.0))
```

Functions with matrices, where all the heavy lifting is done by rules from ChainRules:

```julia
lin(W, x) = sum(W * x)
W, x = randn(3, 2), randn(2)
gradient(lin, W, x)
```

The results agree with Zygote on all of the above (you can check this with `Zygote.gradient`). The gradients with respect to integers are `ZeroTangent()`, where Zygote returns `nothing`.

## Limitations

The AD fails, with an error, on:

* **Broadcasting and splatting.** `sum(tanh.(W * x))` and `sum(map(tanh, W * x))` fail with `Cannot differentiate the builtin call Core._apply_iterate(...)`. `Core._apply_iterate` is the builtin implementing `f(args...)`, and it needs a rule (see the exercises).
* **Mutation.** `y = zeros(2); y[1] = x` fails, since `setindex!` eventually calls the builtin `Core.memoryrefset!`, which has no rule. Supporting mutation requires the reverse pass to restore the overwritten values, which is a substantial piece of work. It is a big part of what Mooncake and Enzyme do.
* **Exceptions.** `try`/`catch` blocks introduce `Expr(:enter)` and `UpsilonNode`/`PhiCNode`, which we do not handle.

It silently gives incomplete gradients for:

* **Closures.** The gradient with respect to captured variables is not computed (functions are in `NonDiff`).
* **Global variables.** These are treated as constants.

And it is **slow**. The tape is a `Vector{Any}`, the gradients are a `Vector{Any}`, and every pullback in the reverse pass is called by dynamic dispatch. For `mypow(1.1, 10)` our `gradient` takes about 4 μs, while `Zygote.gradient` takes about 0.9 μs. (The previous version of this text, which compiled the reverse pass as a separate opaque closure and passed the tape and the trace to it, took about 10 μs.) Mooncake uses a typed stack for every statement (the type of a pullback is known after type inference), stores gradients in SSA form where possible, and runs the compiler's optimizer on the generated code. These optimizations are where most of the complexity of a production AD system lies.

## Exercises

1. **Splatting.** Write a rule for `Core._apply_iterate(iterate, f, args...)`. In the forward pass, flatten the tuples in `args` into a single tuple `flat` and call `ad_rrule(f, flat...)`. In the pullback, split the gradients back into tuples of the original lengths. With this rule, does `sum(tanh.(W * x))` work?
2. **Typed tape.** Instead of a single `Vector{Any}`, create one stack per active statement, typed by the inferred return type of `ad_rrule` (you can infer it with `Base.return_types`). How much faster is `mypow`?
3. **Skipping zeros.** If the gradient of a statement is `ZeroTangent()` (e.g. in the branch not taken), the pullback does not need to be called. Add this optimization to the reverse pass. Be careful: the pullback still has to be popped from the tape.
4. **Debuginfo of the reverse pass.** The reverse pass has no line information. Using [debuginfo.md](debuginfo.md), make every statement of the reverse block point to the line of the original statement it differentiates. Check that an error thrown by a pullback shows the correct line in the stacktrace.
5. **Typed pullback.** Declare the signature of the pullback closure as `Tuple{R}`, where `R` is the type of the output, as in the lecture. Which of the examples above stop working, and why? How would you fix it?
