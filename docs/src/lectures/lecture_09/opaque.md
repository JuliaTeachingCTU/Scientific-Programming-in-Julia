An ordinary closure: its type determines its code

mk_ord(z, k) = k == 1 ? (Δ -> Δ * z) : (Δ -> Δ + z)

Julia lowers each anonymous function to its own struct with a field z. Its body becomes a method of that struct, registered in the global method table:

typeof(mk_ord(2.0, 1)) = var"#mk_ord##0#mk_ord##1"{Float64}   # Δ * z
typeof(mk_ord(2.0, 2)) = var"#mk_ord##2#mk_ord##3"{Float64}   # Δ + z

The two closures have different types, and each type has exactly one method. When the compiler compiles call(c, x) = c(x) + 1.0 and knows the type of c, it uses dispatch to look up the method, reads its body, infers the result type, and inlines it:

1 ─ %1 = Core.getfield(c, :z)::Float64
│   %2 = Base.mul_float(x, %1)::Float64      ← the body of the closure, inlined
│   %3 = Base.add_float(%2, 1.0)::Float64
└──      return %3

This is what "the compiler looks inside" means: from the closure's type it gets to the closure's code.

An opaque closure: the code is stored in the object

mk_op(z, k) = k == 1 ? @opaque(Δ -> Δ * z) : @opaque(Δ -> Δ + z)

typeof(o1) = Core.OpaqueClosure{Tuple{Any}, Any}
typeof(o2) = Core.OpaqueClosure{Tuple{Any}, Any}      # same type, different code
fieldnames  = (:captures, :world, :source, :invoke, :specptr)

Both closures have the same type, and the type says only "takes one argument, returns Any". The code is not a method in the method table. It's carried inside the object: source is the body, and invoke/specptr point to compiled machine code. It works like a function pointer in C. So when the compiler compiles call(o1, x), it knows the type but not the code, and it can't infer or inline anything:

1 ─ %1 = dynamic (c)(x)::Any            ← jump through the pointer stored in the object
│   %2 = dynamic (%1 + 1.0)::Any
└──      return %2

So "the compiler does not look inside when called" means: at the call site the compiler treats the closure as a black box. It calls it through the pointer it holds, and all it knows about the result is the return type in the type parameter (here Any).

One exception: when the compiler sees the closure being created

function inside(z, x)
    o = @opaque Δ -> Δ * z
    o(x) + 1.0
end

1 ─ %1 = Base.mul_float(x, z)::Float64      ← inlined after all
│   %2 = Base.add_float(%1, 1.0)::Float64
└──      return %2

Within the function where the closure is created, the compiler sees the :new_opaque_closure statement, so it knows the closure's code. Inference tracks it as a PartialOpaque. Once the closure is returned from that function, that knowledge is lost and the closure becomes opaque. That's exactly what we see in the lecture:
- the pullback's body is perfectly optimized, because it's compiled for the captured values when the closure is created;
- the caller of fwd only sees OpaqueClosure{Tuple{Any}}, so every call to the pullback is dynamic.

Why we want this anyway

Code stored in an object rather than in the method table is exactly what generating code at run time needs:
- No new methods. Creating a closure from IRCode or a CodeInfo adds no method to the global method table. That means no world-age problems, nothing in existing code gets invalidated, and no generated functions are needed.
- The world age is fixed at creation. The closure uses the definitions that existed when it was created. That's also why the compiler is allowed to optimize the captured values aggressively.
- Any code can be wrapped. An ordinary closure's body must be written in the source. An opaque closure's body can be any code we construct.

A one-sentence version for the lecture: an ordinary closure is a struct whose type selects its code through dispatch, so the compiler can follow it; an opaque closure carries its code inside the object, like a function pointer, so the compiler only knows its argument and return types unless it saw the closure being created.


What an opaque closure knows about types

An OpaqueClosure{A, R} does carry types: A is the type of its arguments and R its return type. What it doesn't carry is which code it runs. Typing then plays out in three places.

1. Inside the body: typed, but only for the declared signature. An ordinary method is compiled separately for every combination of argument types it's called with. An opaque closure is compiled once, for its declared signature A and the types of the values it captured. With a loop directly in the body:

┌────────────────┬─────────┬─────────────────────────────────────────┐
│   signature    │  time   │                allocated                │
├────────────────┼─────────┼─────────────────────────────────────────┤
│ Tuple{Any}     │ 18.8 µs │ 48 KB (every * and + is a dynamic call) │
├────────────────┼─────────┼─────────────────────────────────────────┤
│ Tuple{Float64} │ 0.39 µs │ 80 B                                    │
└────────────────┴─────────┴─────────────────────────────────────────┘

2. At the call site: type-stable, but not inlined. If R is concrete, the caller knows the result type: c(x) + 1.0 is inferred as Float64. The call still goes through the pointer stored in the object, so nothing gets inlined. It cost about 14 ns and 80 bytes per call, against 1.5 ns for an ordinary closure that is inlined.

3. Leaving the function that creates it: the return type is lost. A function that returns an opaque closure is inferred as OpaqueClosure{Tuple{Float64}}, without R, even though the object itself at run time has R = Float64. Anyone calling the pullback returned by fwd therefore makes a dynamic call. A function barrier fixes this: pass the pullback to a small function that calls it, and dispatch specializes on the actual type.

What this means for the lecture

The lecture declares the pullback as sig = Tuple{Any}. Because of point 1, its compiled body treats Δ as Any, and each call to a ChainRules pullback is dynamic. The @code_typed pullback(1.0) listing in the lecture, the one with ten float operations, is code inferred for a Float64 argument. That isn't the code that runs. I'll correct that.

The fix is one line: the pullback's argument has the type of f's output, which code_ircode already returns as its second value (ir, R = only(Base.code_ircode(...))), so we declare sig = Tuple{R}. Gradient of foo, Julia 1.13:

┌─────────────────────────────────────────────────────┬─────────┬──────────┬──────────┐
│                       version                       │ forward │ pullback │ gradient │
├─────────────────────────────────────────────────────┼─────────┼──────────┼──────────┤
│ lecture_v4.md as written (Tuple{Any})               │ 184 ns  │ 108 ns   │ 493 ns   │
├─────────────────────────────────────────────────────┼─────────┼──────────┼──────────┤
│ pullback declared as Tuple{R}                       │ 31 ns   │ 4–14 ns  │ 222 ns   │
├─────────────────────────────────────────────────────┼─────────┼──────────┼──────────┤
│ … plus a function barrier when calling the pullback │         │          │ 66 ns    │
├─────────────────────────────────────────────────────┼─────────┼──────────┼──────────┤
│ Zygote                                              │         │          │ 6 ns     │
└─────────────────────────────────────────────────────┴─────────┴──────────┴──────────┘

The same change gives similar numbers on 1.14-beta1 (130 ns, or 78 ns with the barrier).

One trap. I also tried declaring the forward closure's return type precisely (the concrete type of (y, pullback)). It became slower: 1.95 µs instead of 184 ns. The profile shows jl_interpret_opaque_closure: Julia stopped compiling the closure and ran it in its interpreter. My explanation is that the declared return type was narrower than what inference can prove for that body, but I haven't confirmed that in Julia's source. In practice: declare argument types precisely, and leave the return type as Any unless you can be sure inference will prove it.

Does it suck?

It's a real cost, and it's how opaque closures are designed: the body is optimized, the boundary isn't. The three systems deal with it differently:
- Zygote doesn't use opaque closures at all. Its pullback is an ordinary struct whose call is a generated function, so the compiler can see through everything and inline it. That's why it reaches 6 ns here, and why it depends on generated functions.
- Mooncake uses opaque closures but builds them from typed IRCode with a return type it computed in advance (misty_closure(fwd_ret_type, fwd_ir, …) in build_derived_rrule). It also gives each rule a concrete type ahead of time (rule_type, LazyDerivedRule{sig, Trule}), so calls between rules stay type-stable. It pays only the cost of a call that isn't inlined at each rule boundary, and in exchange it avoids generated functions.
- Diffractor had the problem we have here, which is part of why its performance was uneven.



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
