```@setup lec08
using Plots
```

<!-- I have once seen a nice tutorial / lecture on AD by Matthew Johnson, but I cannot find it anymore.  -->
# Automatic Differentiation

## Motivations to study an automatic differentiation
- It it support a lot of modern machine learning by allowing quick differentiation of complex mathematical functions. Recall that 1st order optimization methods are ubiquituous in finding parameters of functions.
- AD is interesting to study from the implementation perspective. There are different takes on it with different trade-offs and Julia offers many implementations (some of them of course not maintained anymore).
- We (authors of this course) believe that it is good to understand (at least roughly), how the methods work in order to use them effectively in our research.
- Julia is unique in the effort separating definitions of AD rules from engines that use those rules to perform the AD. This allows authors of libraries to add new rules that would be compatible 

## Theory
The differentiation is routine process, as most of the time we can break down complex functions into small pieces that we know, how to differentiate and from that to assemble the gradient of the complex function back. Thus, the essential piece is the differentiation of the composed function, which is called chainrule.

Specifically, for a composed function ``f: \mathbb{R}^n \rightarrow \mathbb{R}^m``

``f(x) = f_1(f_2(f_3(\ldots f_n(x))))`` 

the gradient with respect to `x` is equal to

``\frac{\partial f}{\partial x} = \frac{f_1}{\partial y_1} \times \frac{f_2}{\partial y_2} \times \ldots \times \frac{f_n}{\partial y_n}``, 

where

``y_i = f_{i+1}(f_{i+1}(\ldots f_n(x)))`` and ``y_n = x``.

How ``\frac{f_i}{\partial y_i}`` looks like? If ``f_i: \mathbb{R} \rightarrow \mathbb{R}``, then ``\frac{f_i}{\partial y_i} \in \mathbb{R}`` is a real number ``\mathbb{R}``. On the other hand if ``f_i: \mathbb{R}^{m_i} \rightarrow \mathbb{R}^{n_i}``, then ``\frac{f_i}{\partial y_i} \in \mathbb{R}^{n_i,m_i}`` is a matrix and the computation of gradient ``\frac{\partial f}{\partial x}`` boils down to matrix multiplication. Denoting `i`-th Jacobian as ``J_i = \frac{f_i}{\partial y_i}`` it holds that ``\frac{\partial f}{\partial x} = J_1 \times J_2 \times \ldots \times J_n``. 

Matrix multiplication is generally expensive, as it has at least ``O(n^{2.3728596})`` (where the devil is hidden in the ``O(n)``). The order in which the Jacobians are multiplied has therefore a profound effect on the complexity of the AD engine. While determining the optimal order of multiplication of sequence of matrices is costly (it can be solved using dynamic programming), it is interesting to investigate two cases.

In the first case, we multiply Jacobians from right to left, i.e. as 
``J_1 \times (J_2 \times ( \ldots \times (J_{n-1}) \times J_n))))``
or the other way around from left to right
``((((J_1 \times J_2) \times J_3) \times \ldots ) \times J_n``
The first approach has the advantage that if the input dimension of ``f`` is smaller than the output dimension (``n < m`` where be reminded that ``f: \mathbb{R}^n \rightarrow \mathbb{R}^m``), while the latter has the advantage in the oposite case where output dimension is smaller that the input one. In deep learning, one mostly do the opposite case. Also notice that for `f` of certain structures, it pays-off to do a mixed-mode AD, where some partse are done using forward diff and some parts using reverse diff. 

### Forward mode
Initialize the jacobian of ``y_n`` with respect to ``x`` to an identity matrix, because as we have stated above ``y_n = x``, i.e. ``\frac{\partial y_n}{\partial x} = \mathbb{I}``.
Iterate `i` from `n` down to `1` as
- calculate the next intermediate ourput as ``y_{i-1} = f_i{y_i}`` 
- calculate Jacobian ``J_i = \frac{f_i}{\partial y_i}`` at point ``y_i``
- *push forward* the gradient as ``\frac{\partial y_{i-1}}{\partial x} = J_i \times \frac{\partial y_n}{\partial x}``

Notice that 
- on the very end, we are left with `y = y_0` and with ``\frac{\partial y_0}{\partial x}``, which is the gradient we wanted to calculate;
- if `y` is a scalar, then ``\frac{\partial y_0}{\partial x}`` is a matrix with single row (but in usual notation, it should be single column, right?)
- the jacobian and the output of the function is calculated in one sweep.

The above is an idealized computation. The real implementation is a bit different, as we will see later.

### Reverse mode
In reverse mode, the computation of the gradient follow the oposite order. That means.
We initialize the Jacobian by ``\frac{\partial y}{\partial y_0},`` which is again an identity matrix, and perform the computation of jacobians and multiplications in the opposite order. The problem is that to calculate jacobian ``J_i``, we need to know ``y_i``. Therefore the reverse mode diff requires two passes over the computation graph.

1. Forward pass: iterate `i` from `n` down to `1` as
    - calculate the next intermediate output as ``y_{i-1} = f_i(y_i)`` 

2. Backward pass: iterate `i` from `1` down to `n` as
    - calculate Jacobian ``J_i = \frac{f_i}{\partial y_i}`` at point ``y_i``
    - *pull back* the gradient as ``\frac{\partial y_0}{\partial y_{i}} = \frac{\partial y_0}{\partial y_{i-1}} \times J_i``
    Notice that we actually cannot compute the backward pass, as we do not know ``y_i`` at the moment of computing the `J_i`. Therefore the backward pass needs to be preceeded by forward pass, where we calculate ``y_i`` as follows


The fact that we need to store intermediate outs has a huge impact on the memory requirements. Therefore when we have been talking few lectures ago that we should avoid excessive memory allocations, here we have an algorithm where the excessive allocation is by design. 

## Let's work an example 

### `n` to `1` function 

### `1` to `n` function 

## What are the tricks that we can use?
- we define custom rules over large functional blocks. For example while we can auto-grad (in theory) matrix product, it is much more efficient to define make a matrix multiplication as one large function, for which we define jacobians (not to say that by doing so, we can dispatch on Blas)
- **Invertible functions** When we are differentiating invertible functions, we can calculate intermediate outputs from the 
- **Checkpointing** We can store the intermediate ouputs only sometime and performed a small forward while doint backward
- ** Implicit functions** Differentiating through solvers is possible, since they are mostly iterative. But, using the mathematical equality is easier.

!!! info
    Reverse mode AD was first published in 1976 by Seppo Linnainmaa, a finnish computer scientist. It was popularized in the end of 80s when applied to training multi-layer perceptrons, which gave rise to the famous **backpropagation** algorithm, which is a special case of reverse mode AD.

### Implementation
- TensorFlow with explicit graph vs. PyTorch with tape / Wengert list

### How to test
A tricky example with odd function


### ChainRules
- Why you want to have it
- Syntax
- Structural vs Natural gradient

## Why custom rules
- We need them for speed
- They are needed for numerical stability


## Theory

### Forward

### Reverse

### How to test
A tricky example with odd function

### ChainRules
- Why you want to have it
- Syntax
- Structural vs Natural gradient

# 1. Introduction

**_Automatic differentiation_** (AD) is the at the core of one of the most widely
used local optimization methods: **_Gradient Descent_** (GD), which finds a local
optimum of a differentiable function $f(\bm w) : \mathbb R^N \rightarrow
\mathbb R$

```math
\bar{\bm w} = \bm w - \eta \frac{\partial f(\bm w)}{\partial \bm w}.
```

Computing the gradient of $f$ manually is tedious (think about your fancy new
neural network layer). You probably never want to do it by hand, because it
takes time, is not fun, and you'll make mistakes (because its not fun). Still
it is good to have a mental model of how your AD library works, just like its
good to know how a compiler does its magic. For example, there are cases in
which different AD systems are faster (forward vs. reverse), but more on that
later.

## Chain Rules & Jacobians

Assume we want to derive the function $f$ that is composed of three functions,
where $f_1:\mathbb R^N\rightarrow \mathbb R^M$, $f_2:\mathbb R^M\rightarrow
\mathbb R^M$, and $f_3:\mathbb R^M\rightarrow \mathbb R^L$
```math
\bm y = f(\bm x) = (f_1 \circ f_2 \circ f_3)(\bm x).
\qquad\rightarrow\qquad
\begin{matrix} \bm y_1 = f_1(\bm x) \\ \bm y_2 = f_2(\bm y_1) \\ \bm y_3 = f(\bm y_2) \end{matrix}
```
If we know the **_Jacobian_** (matrix of partial derivatives)
```math
\bm J = \{J\}_{ij} = \left\{ \frac{\partial y_i}{\partial x_j} \right\}_{ij}
```
we only need to apply the chain rule
```math
\frac{d\bm y}{d\bm x} = \frac{d\bm y_1}{d\bm x}\frac{d\bm y_2}{d\bm y_1}\frac{d\bm y_3}{d\bm y_2}
```
to obtain the Jacobian of $f$. Algorithmically, we have two different
possibilities of computing $\bm J$: starting from the beginning (at $\frac{d\bm
y_1}{d\bm x}$) going **_forward_**, or starting from the end (at
$\frac{d\bm y_3}{d\bm y_2})$ going in **_reverse_** order. Depending on where we
start, we have to perform a different number of multiplications.

```math
\begin{matrix}\frac{d\bm y}{d\bm x}     \\ N \times M \end{matrix}
    \qquad \rightarrow \qquad
\begin{matrix}\frac{d\bm y_1}{d\bm x}   \\ N \times L  \end{matrix}
    \qquad
\begin{matrix}\frac{d\bm y_1}{d\bm y_2} \\ L \times L  \end{matrix}
    \qquad
\begin{matrix}\frac{d\bm y_2}{d\bm y_3} \\ L \times M  \end{matrix}
```
The result on the left always has the same size $N\times M$, but for the
forward pass, $N\cdot L\cdot L + N\cdot L\cdot M$ multiplications have to be
computed, while for the backward pass we get $M\cdot L\cdot L + M\cdot L\cdot
N$.  Hence, if $N < M$ we should be using forward mode, and for $N>M$ reverse
mode.  In practice we have to consider additional implementation details when
it comes to performance, which we will discuss in the last section.


# 2. Forward Mode

In this section we will demonstrate how to implement a scalar forward mode AD.
In forward mode we need to keep track of the derivative at each computation
step in order to find the derivative of our complicated function $f$.  This can
be elegantly realized with a custom number type called the **_dual number_**
(for vector valued functions they are called *hyperduals*).  Conceptually, they
are very similar to complex numbers, but instead of the imaginary number $i$
dual numbers use $\epsilon$ in its second component:
```math
x = v + \dot v \epsilon,
```
where $(v,\dot v) \in \mathbb R$, $\epsilon\neq 0$, and $\epsilon^2=0$ (instead
of $i^2=-1$).  The second component $\dot v$ can be used to keep track of the
derivative at each (simple) computational step.  This becomes clear
when considering the Taylor expansion around a dual number
```math
f(v+\dot v \epsilon) = \sum_{n=0}^\infty \frac{f^n(v)\dot v^n\epsilon^n}{n!}
  = f(v) + f'(v)\dot v\epsilon,
```
where all higher order terms can be dropped because $\epsilon^n=0$ for $n>1$.
This means we just need to know the the derivatives of the simple
subexpressions in the WL and we will end up with the derivative of $f$ in the
second component of $x$. 

To demonstrate the simplicity of this approach we can compute the derivative of
the **_Babylonian Square Root_** (an algorithm to compute $\sqrt x$):

```@repl lec08
babysqrt(x, t=(1+x)/2, n=10) = n==0 ? t : babysqrt(x, (t+x/t)/2, n-1)
babysqrt(2)
```
All we need is to define a `Dual` number type with a value component `x`, a
derivative component `d`, and overload the functions `+`, `/`.  In Julia, this
reads:
```@example lec08
struct Dual{T<:Number}
    x::T
    d::T
end

# Componentwise addition: (a+vϵ) + (b+wϵ) = (a+b) + (v+w)ϵ
Base.:+(a::Dual, b::Dual)   = Dual(a.x+b.x, a.d+b.d)
Base.:+(a::Dual, n::Number) = Dual(a.x + n, a.d)
Base.:+(n::Number, a::Dual) = a + n

# Dual Nr. Law: (a+vϵ) / (b+wϵ) = a/b  +  (a/b)'ϵ
# Quotient rule: (a/b)' = (a'b - ab')/b^2
Base.:/(a::Dual, b::Dual)   = Dual(a.x/b.x, (a.d*b.x - a.x*b.d)/b.x^2)
Base.:/(n::Number, a::Dual) = Dual(a.x/n, -n*a.d/a.x^2)
Base.:/(a::Dual, n::Number) = Dual(a.x/n, a.d/n)
nothing # hide
```
Now we can define a general function that computes the derivative of a function
`f`. We just have to call `f` with a dual number with derivative component `d=1`
(because $\frac{dx}{dx}=1$).
```@example lec08
forward(f::Function, x::Number) = f(Dual(x,1.0)).d
nothing # hide
```
To demonstrate that it acutally works we can compare the analytic solution to
our AD version:
```math
f(x) = \sqrt{x} \qquad f'(x) = \frac{1}{2\sqrt{x}}
```
```@repl lec08
forward_dsqrt(x) = forward(babysqrt,x)
analytc_dsqrt(x) = 1/(2babysqrt(x))
forward_dsqrt(2.0)
analytc_dsqrt(2.0)
```
```@example lec08
plot(0.0:0.01:2, babysqrt, label="f(x) = babysqrt(x)", lw=3)
plot!(0.1:0.01:2, analytc_dsqrt, label="Analytic f'", ls=:dot, lw=3)
plot!(0.1:0.01:2, forward_dsqrt, label="Dual Forward Mode f'", lw=3, ls=:dash)
```
---
### Takeaways
1. Forward mode $f'$ is obtained simply by pushing a `Dual` through `babysqrt`
2. We only need to **_overload_** a few **_operators_** for forward mode AD to
   work on **_any function_**
3. For vector valued function we can use **_Hyperduals_**
---

The equivalent in Python looks like this:
```julia
class Dual:
    def __init__(self, x, d):
        self.x = x
        self.d = d

    def __add__(self, a):
        if isinstance(a, Dual):
            return Dual(self.x+a.x, self.d+a.d)
        else:
            return Dual(self.x+a, self.d)

    def __radd__(self, a):
        return self + a

    def __truediv__(self, b):
        if isinstance(b, Dual):
            return Dual(self.x / b.x, (self.d*b.x - self.x*b.d) / (b.x**2))
        else:
            return Dual(self.x/b, self.d/b)

    def __rtruediv__(self, b):
        if isinstance(b, Dual):
            return self.__truediv__(b)
        else:
            return Dual(self.x / b, -b*self.d / self.x**2)

    def __repr__(self):
        return f"Dual({self.x}, {self.d})"

def forward(f,x):
    return f(Dual(x,1)).d

```



# 3. Reverse Mode (Backpropagation)

Reverse Mode AD is at the core of the advances made in Machine Learning.
Neural Networks are trained by optimizing a scalar loss function (with many
inputs) and therefore Reverse Mode tends to be faster.
Remember our vector valued function
```math
f(\bm x) = (f_1 \circ f_2 \circ f_3)(\bm x).
\qquad\rightarrow\qquad
\begin{matrix} \bm y_1 = f_1(\bm x) \\ \bm y_2 = f_2(\bm y_1) \\ \bm y_3 = f(\bm y_2) \end{matrix}
```
with the Jacobian
```math
\frac{\partial f(\bm x)}{\partial \bm x}
 = \frac{\partial f_1(\bm x)}{\partial \bm x}
   \frac{\partial f_2(y_1)}{\partial \bm y_1}
   \frac{\partial f_3(\bm y_2)}{\partial \bm y_2}
   \frac{\partial \bm y_3}{\partial \bm y_3}.
```
Now we want to start computing the gradient starting from the last term for
which we need both $f_3$ and its input $\bm y_2$ that produced $\bm y_3$.  This means
that we have to keep track of functions and intermediate values ($\bm y_1$, $\bm
y_2$) during the forward pass. This can be implemented by a `rrule` (*reverse rule*) which
returns the output itself along with a function that computes the gradient of
the input called (**_pullback_**). The argument to the pullback (`Δ`) represents the gradient of the
previously evaluated function ("on the right side" - in case of $\frac{\partial f_3}{\partial\bm y_2}$
this would be `Δ`$\,=\frac{\partial\bm y_3}{\partial\bm y_3}=1$).

```julia
rrule(::typeof(f1), x) = f1(x), Δ -> Δ*f1'(x)
```

## Tracing-based AD

Libraries like **_PyTorch_**
solve this problem by building up a *tape* or *computation graph* with
a new type that tracks inputs, outputs, and functions of the computation.  In
Julia this is approach realized in **_ReverseDiff.jl_**.  It is based
on a `Tracked` type which contains a the data (i.e. value) of a number, and
a tracker which holds information about how the value was computed (i.e.
function + arguments) and a placeholder for the gradient.  The arguments that
are stored in the tracker can again be a `Tracked` which results in a tree
structure that represents the computation graph.

Tracing-based AD is quite easy to implement but has some disadvantages.  For
example, loops result in deeper and deeper computation graphs, which leads to
increased memory usage negatively impacting performance. Functions that should
be differentiated have to be written such that they accept the `Tracked` types.

## Source Code Transformation

The most recent approach to Reverse Mode AD is **_Source-to-Source_**
transformation adopted by packages like **_JAX_** and **_Zygote.jl_**.
Transforming code promises to eliminate the problems of tracing-based AD.
`Tracked` types are not needed anymore, which reduces memory usage, promising
significant speedups. Additionally, the reverse pass becomes a *compiler
problem*, which makes it possible to leverage highly optimized compilers like
LLVM.

Source-to-source AD uses meta-programming to produce `rrule`s for any function
that is a composition of available `rrule`s. The code for `foo`
```@example lec08
foo(x) = h(g(f(x)))

f(x) = x^2
g(x) = sin(x)
h(x) = 5x
nothing # hide
```
is transformed into
```julia eval=false
function rrule(::typeof(foo), x)
    a, Ja = rrule(f, x)
    b, Jb = rrule(g, a)
    y, Jy = rrule(h, b)

    function dfoo(Δy)
        Δb = Jy(Δy)
        Δa = Jb(Δb)
        Δx = Ja(Δa)
        return Δx
    end
    
    return y, dfoo
end
```
For this simple example we can define the three `rrule`s by hand:
```@example lec08
rrule(::typeof(f), x) = f(x), Δ -> 2x*Δ
rrule(::typeof(g), x) = g(x), Δ -> cos(x)*Δ
rrule(::typeof(h), x) = h(x), Δ -> 5*Δ
```
Remember that this is a very artificial example. In real AD code you would
overload functions like `+`, `*`, etc, such that you don't have to define a
`rrule` for something like `5x`.

In order to transform our functions safely we will make use of `IRTools.jl`
(*Intermediate Representation Tools*) which provide some convenience functions
for inspecting and manipulating code snippets. The IR for `foo` looks like this:
```@example lec08
using IRTools: @code_ir, evalir
ir = @code_ir foo(2.)
```
```@setup lec08
msg = """
ir = 1: (%1, %2)                 ## rrule(foo, x)
       %3 = Main.f(%2)           ##   a = f(x)
       %4 = Main.g(%3)           ##   b = g(a)
       %5 = Main.h(%4)           ##   y = h(b)
       return %5                 ##   return y
"""
```
Variable names are replaced by `%N` and each function gets is own line.
We can evalulate the IR (to actually run it) like this
```@example lec08
evalir(ir, nothing, 2.)
```
As a first step, lets transform the function calls to `rrule` calls.  For
this, all we need to do is iterate through the IR line by line and replace each
statement with `(Main.rrule)(Main.func, %N)`, where `Main` just stand for the
gobal main module in which we just defined our functions.
But remember that the `rrule` returns
the value `v` *and* the pullback `J` to compute the gradient. Just
replacing the statements would alter our forward pass. Instead we can insert
each statement *before* the one we want to change. Then we can replace the the
original statement with `v = rr[1]` to use only `v` and not `J` in the
subsequent computation.
```@example lec08
using IRTools
using IRTools: xcall, stmt

xgetindex(x, i...) = xcall(Base, :getindex, x, i...)

ir = @code_ir foo(2.)
pr = IRTools.Pipe(ir)

for (v,statement) in pr
    ex = statement.expr
    rr = xcall(rrule, ex.args...)
    # pr[v] = stmt(rr, line=ir[v].line)
    vJ = insert!(pr, v, stmt(rr, line = ir[v].line))
    pr[v] = xgetindex(vJ,1)
end
ir = IRTools.finish(pr)
#
#msg = """
#ir = 1: (%1, %2)                          ## rrule(foo, x)
#       %3 = (Main.rrule)(Main.f, %2)      ##   ra = rrule(f,x)
#       %4 = Base.getindex(%3, 1)          ##   a  = ra[1]
#       %5 = (Main.rrule)(Main.g, %4)      ##   rb = rrule(g,a)
#       %6 = Base.getindex(%5, 1)          ##   b  = rb[1]
#       %7 = (Main.rrule)(Main.h, %6)      ##   ry = rrule(h,b)
#       %8 = Base.getindex(%7, 1)          ##   y  = ry[1]
#       return %8                          ##   return y
#"""
#println(msg)
```
Evaluation of this transformed IR should still give us the same value
```@example lec08
evalir(ir, nothing, 2.)
```

The only thing that is left to do now is collect the `Js` and return
a tuple of our forward value and the `Js`.
```@example lec08
using IRTools: insertafter!, substitute, xcall, stmt

xtuple(xs...) = xcall(Core, :tuple, xs...)

ir = @code_ir foo(2.)
pr = IRTools.Pipe(ir)
Js = IRTools.Variable[]

for (v,statement) in pr
    ex = statement.expr
    rr = xcall(rrule, ex.args...)  # ex.args = (f,x)
    vJ = insert!(pr, v, stmt(rr, line = ir[v].line))
    pr[v] = xgetindex(vJ,1)

    # collect Js
    J = insertafter!(pr, v, stmt(xgetindex(vJ,2), line=ir[v].line))
    push!(Js, substitute(pr, J))
end
ir = IRTools.finish(pr)
# add the collected `Js` to `ir`
Js  = push!(ir, xtuple(Js...))
# return a tuple of the last `v` and `Js`
ret = ir.blocks[end].branches[end].args[1]
IRTools.return!(ir, xtuple(ret, Js))
ir
#msg = """
#ir = 1: (%1, %2)                          ## rrule(foo, x)
#       %3 = (Main.rrule)(Main.f, %2)      ##   ra = rrule(f,x)
#       %4 = Base.getindex(%3, 1)          ##   a  = ra[1]
#       %5 = Base.getindex(%3, 2)          ##   Ja = ra[2]
#       %6 = (Main.rrule)(Main.g, %4)      ##   rb = rrule(g,a)
#       %7 = Base.getindex(%6, 1)          ##   b  = rb[1]
#       %8 = Base.getindex(%6, 2)          ##   Jb = rb[2]
#       %9 = (Main.rrule)(Main.h, %7)      ##   ry = rrule(h,b)
#       %10 = Base.getindex(%9, 1)         ##   y  = ry[1]
#       %11 = Base.getindex(%9, 2)         ##   Jy = ry[2]
#       %12 = Core.tuple(%5, %8, %11)      ##   Js = (Ja,Jb,Jy)
#       %13 = Core.tuple(%10, %12)         ##   rr = (y, Js)
#       return %13                         ##   return rr
#"""
#println(msg)
```
The resulting IR can be evaluated to the forward pass value and the Jacobians:
```@repl lec08
(y, Js) = evalir(ir, foo, 2.)
```
To compute the derivative given the tuple of `Js` we just need to compose them
and set the initial gradient to one:
```@repl lec08
reduce(|>, Js, init=1)  # Ja(Jb(Jy(1)))
```
The code for transforming the IR as described above looks like this.
```@example lec08
function transform(ir, x)
    pr = IRTools.Pipe(ir)
    Js = IRTools.Variable[]
    
    # loop over each line in the IR
    for (v,statement) in pr
        ex = statement.expr
        # insert the rrule
        rr = xcall(rrule, ex.args...)  # ex.args = (f,x)
        vJ = insert!(pr, v, stmt(rr, line = ir[v].line))
        # replace original line with f(x) from rrule
        pr[v] = xgetindex(vJ,1)
    
        # save jacobian in a variable
        J = insertafter!(pr, v, stmt(xgetindex(vJ,2), line=ir[v].line))
        # add it to a list of jacobians
        push!(Js, substitute(pr, J))
    end
    ir = IRTools.finish(pr)
    # add the collected `Js` to `ir`
    Js  = push!(ir, xtuple(Js...))
    # return a tuple of the foo(x) and `Js`
    ret = ir.blocks[end].branches[end].args[1]
    IRTools.return!(ir, xtuple(ret, Js))
    return ir
end

xgetindex(x, i...) = xcall(Base, :getindex, x, i...)
xtuple(xs...) = xcall(Core, :tuple, xs...)
nothing # hide
```
Now we can write a general `rrule` that can differentiate any function
composed of our defined `rrule`s
```@example lec08
function rrule(f, x)
    ir = @code_ir f(x)
    ir_derived = transform(ir,x)
    y, Js = evalir(ir_derived, nothing, x)
    df(Δ) = reduce(|>, Js, init=Δ)
    return y, df
end


reverse(f,x) = rrule(f,x)[2](one(x))
nothing # hide
```
Finally, we just have to use `reverse` to compute the gradient
```@example lec08
plot(-2:0.1:2, foo, label="f(x) = 5sin(x^2)", lw=3)
plot!(-2:0.1:2, x->10x*cos(x^2), label="Analytic f'", ls=:dot, lw=3)
plot!(-2:0.1:2, x->reverse(foo,x), label="Dual Forward Mode f'", lw=3, ls=:dash)
```

---
- Efficiency of the forward pass becomes essentially a compiler problem
- If we define specialized rules we will gain performance
---

# Performance Forward vs. Reverse

This section compares the performance of three different, widely used Julia AD
systems `ForwardDiff.jl` (forward mode), `ReverseDiff.jl` (tracing-based
reverse mode), and `Zygote.jl` (source-to-source reverse mode), as well as JAX
forward/reverse modes.

As a benchmark function we can compute the Jacobian of $f:\mathbb R^N
\rightarrow \mathbb R^M$ with respect to $\bm x$.
In the benchmark we test various different values of $N$ and $M$ to show the
differences between the backends.
```math
f(\bm x) = (\bm W \bm x + \bm b)^2
```

```@setup lec08
using DataFrames
using DrWatson
using Glob


julia_res = map(glob("julia-*.txt")) do fname
    d = parse_savename(replace(fname, "julia-"=>""))[2]
    @unpack N, M = d
    lines = open(fname) |> eachline
    map(lines) do line
        s = split(line, ":")
        backend = s[1]
        time = parse(Float32, s[2]) / 10^6
        (backend, time, "$(N)x$(M)")
    end
end

jax_res = map(glob("jax-*.txt")) do fname
    d = parse_savename(replace(fname, "jax-"=>""))[2]
    @unpack N, M = d
    lines = open(fname) |> eachline
    map(lines) do line
        s = split(line, ":")
        backend = s[1]
        time = parse(Float32, s[2]) * 10^3
        (backend, time, "$(N)x$(M)")
    end
end

res = vcat(julia_res, jax_res)

df = DataFrame(reduce(vcat, res))
df = unstack(df, 3, 1, 2)
ns = names(df)
ns[1] = "N x M"
rename!(df, ns)
df = DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])

ns = df[1,:] |> values |> collect
rename!(df, ns)
```
```@example lec08
df[2:end,:] # hide
```

# TODO

* show constant gradient of linear function in LLVM
* jacobians explaination: show figures with many-to-ont and one-to-many
* [simplest viable implementation](https://juliadiff.org/ChainRulesCore.jl/dev/autodiff/operator_overloading.html#ReverseDiffZero)

# Sources

* Mike Innes' [diff-zoo](https://github.com/MikeInnes/diff-zoo)
* [Write Your Own StS in One Day](https://blog.rogerluo.me/2019/07/27/yassad/)
* [Zygote.jl Paper](https://arxiv.org/pdf/1810.07951.pdf)
  and [Zygote.jl Internals](https://fluxml.ai/Zygote.jl/dev/internals/)
* Keno's [Talk](https://www.youtube.com/watch?v=mQnSRfseu0c&feature=youtu.be)
* Chris' [Lecture](https://mitmath.github.io/18337/lecture11/adjoints)
