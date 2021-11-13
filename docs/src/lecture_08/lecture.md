```@setup lec08
using Plots
```
<!-- I have once seen a nice tutorial / lecture on AD by Matthew Johnson, but I cannot find it anymore.  -->
# Automatic Differentiation

## Motivation
- It supports a lot of modern machine learning by allowing quick differentiation of complex mathematical functions. The 1st order optimization methods are ubiquituous in finding parameters of functions (not only in  deep learning).
- AD is interesting to study from the implementation perspective. There are different takes on it with different trade-offs and Julia offers many implementations (some of them are not maintained anymore).
- We (authors of this course) believe that it is good to understand (at least roughly), how the methods work in order to use them effectively in our work.
- Julia is unique in the effort separating definitions of AD rules from AD engines that use those rules to perform the AD. This allows authors of generic libraries to add new rules that would be compatible with many frameworks.

## My Theory

The differentiation is routine process, as most of the time we break complicated functions down into small pieces that we know, how to differentiate and from that to assemble the gradient of the complex function back. Thus, the essential piece is the differentiation of the composed function ``f: \mathbb{R}^n \rightarrow \mathbb{R}^m``

``f(x) = f_1(f_2(f_3(\ldots f_n(x)))) = (f_1 \circ f_2 \circ \ldots \circ f_n)(x)`` 

which is computed by chainrule. Before we dive into the details, let's define the notation, which for the sake of clarity needs to be precise. The gradient of function `f(x)` with respect to `x` at point `x₀` will be denoted as 
``\left.\frac{\partial f}{\partial x}\right|_{x^0}``

For a composed function ``f(x)`` the gradient with respect to ``x`` at point ``x_0`` is equal to
```math
\left.\frac{\partial f}{\partial x}\right|_{x^0} = \left.\frac{f_1}{\partial y_1}\right|_{y_1^0} \times \left.\frac{f_2}{\partial y_2}\right|_{y_2^0} \times \ldots \times \left.\frac{f_n}{\partial y_n}\right|_{y_n^0},
```

where ``y_i`` denotes the input of function ``f_i`` and
```math
\begin{alignat*}{2}
y_i^0 = &\ \left(f_{i+1} \circ \ldots \circ f_n\right) (x^0) \\
y_n^0 = &\ x^0 \\
y_0^0 = &\ f(x^0) \\
\end{alignat*}
```
How ``\left.\frac{f_i}{\partial y_i}\right|_{y_i^0}`` looks like? 
- If ``f_i: \mathbb{R} \rightarrow \mathbb{R}``, then ``\frac{f_i}{\partial y_i} \in \mathbb{R}`` is a real number ``\mathbb{R}`` and we live in a high-school world, where it was sufficient to multiply real numbers.
- If ``f_i: \mathbb{R}^{m_i} \rightarrow \mathbb{R}^{n_i}``, then ``\mathbf{J}_i = \left.\frac{f_i}{\partial y_i}\right|_{y_i^0} \in \mathbb{R}^{n_i,m_i}`` is a matrix with ``m_i`` rows and ``n_i`` columns. The computation of gradient ``\frac{\partial f}{\partial x}`` *theoretically* boils down to 
   1. computing Jacobians ``\left\{\mathbf{J}_i\right\}_{i=1}^n`` 
   2. multiplication of Jacobians as it holds that ``\left.\frac{\partial f}{\partial x}\right|_{y_0} = J_1 \times J_2 \times \ldots \times J_n``. 

The complexity of the computation (at least one part of it) is therefore therefore determined by the  Matrix multiplication, which is generally expensive, as theoretically it has complexity at least ``O(n^{2.3728596}),`` but in practice a little bit more as the lower bound hides the devil in the ``O`` notation. The order in which the Jacobians are multiplied has therefore a profound effect on the complexity of the AD engine. While determining the optimal order of multiplication of sequence of matrices is costly, in practice, we recognize two important cases.

1. Jacobians are multiplied from right to left as  ``J_1 \times (J_2 \times ( \ldots \times (J_{n-1}) \times J_n))))`` which has the advantage when the input dimension of ``f: \mathbb{R}^n \rightarrow \mathbb{R}^m`` is smaller than the output dimension, ``n < m``.
2. Jacobians are multiplied from left to right as ``((((J_1 \times J_2) \times J_3) \times \ldots ) \times J_n`` which has the advantage when the input dimension of ``f: \mathbb{R}^n \rightarrow \mathbb{R}^m`` is larger than the output dimension, ``n < m``.
The ubiquituous in machine learning to minimization of a scalar (loss) function of a large number of parameters. Also notice that for `f` of certain structures, it pays-off to do a mixed-mode AD, where some partse are done using forward diff and some parts using reverse diff. 

### Example
Let's workout an example
```math
z = xy + sin(x)
```
How it maps to the notation we have used above? Particularly, what are ``f_1, f_2, \ldots, f_n`` and the corresponding ``\{y_i\}_{i=1}^n``?
```math
\begin{alignat*}{6}
f_1:&\mathbb{R}^2 \rightarrow \mathbb{R} \quad&f_1(y_1)& = y_{1,1} + y_{1,2}            \quad & y_0 = & (xy + \sin(x))           \\
f_2:&\mathbb{R}^3 \rightarrow \mathbb{R}^2 \quad&f_2(y_2)& = (y_{2,1}y_{1,2}, y_{2,3}) \quad & y_1 = &  (xy, \sin(x))&\\
f_3:& \mathbb{R}^2 \rightarrow \mathbb{R}^3 \quad&f_3(y_3)& = (y_{3,1}, y_{3,2}, \sin(y_{3,1}))  \quad & y_2 =& (x, y, \sin(x))\\
\end{alignat*}
```
The corresponding jacobians are 
```math
\begin{alignat*}{4}
f_1(y_1) & = y_{1,1} + y_{1,2}             \quad & \mathbf{J}_1& = \begin{bmatrix} 1 \\ 1 \end{bmatrix}  \\
f_2(y_2) & = (y_{2,1}y_{1,2}, y_{2,3})     \quad & \mathbf{J}_2& = \begin{bmatrix} y_{2, 2} & 0 \\ y_{2,1} & 0 0 1 \end{bmatrix}\\
f_3(y_3) & = (y_{3,1}, y_{3,2}, \sin(y_{3,1}))     \quad & \mathbf{J}_3 & = \begin{bmatrix} 1 & 0 & \cos(y_{3,1}) \\ 0 & 1 & 0 \end{bmatrix} \\
\end{alignat*}
```
and for the gradient it holds that
```math
\begin{bmatrix} \frac{\partial f(x, y)}{\partial{x}} \\ \frac{\partial f(x,y)}{\partial{y}} \end{bmatrix} = \mathbf{J}_3 \times \mathbf{J}_2 \times \mathbf{J}_1 =  \begin{bmatrix} 1 & 0 & \cos(x) \\ 0 & 1 & 0 \end{bmatrix} \\  \times \begin{bmatrix} y & 0 \\ x & 0 \\ 0 & 1 \end{bmatrix} \times \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} y & \cos(x) \\ x & 0 \end{bmatrix} \times \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} y + \cos(x) \\ x \end{bmatrix}
```

## Calculation of the Forward mode
In theory, we can calculate the gradient using forward mode as follows
Initialize the jacobian of ``y_n`` with respect to ``x`` to an identity matrix, because as we have stated above ``y_n = x``, i.e. ``\frac{\partial y_n}{\partial x} = \mathbb{I}``.
Iterate `i` from `n` down to `1` as
- calculate the next intermediate ourput as ``y_{i-1} = f_i{y_i}`` 
- calculate Jacobian ``J_i = \frac{f_i}{\partial y_i}`` at point ``y_i``
- *push forward* the gradient as ``\frac{\partial y_{i-1}}{\partial x} = J_i \times \frac{\partial y_n}{\partial x}``

Notice that 
- on the very end, we are left with `y = y_0` and with ``\frac{\partial y_0}{\partial x}``, which is the gradient we wanted to calculate;
- if `y` is a scalar, then ``\frac{\partial y_0}{\partial x}`` is a matrix with single row
- the jacobian and the output of the function is calculated in one sweep.

The above is an idealized computation. The real implementation is a bit different, as we will see later.

### Implementation of the forward mode using Dual numbers
Forward modes need to keep track of the output of the function and of the derivative at each computation step in the computation of the complicated function $f$. This can be elegantly realized with a (**dual number**)[https://en.wikipedia.org/wiki/Dual_number], which are conceptually similar to complex numbers, but instead of the imaginary number ``i`` dual numbers use ``\epsilon`` in its second component:
```math
x = v + \dot v \epsilon,
```
where ``(v,\dot v) \in \mathbb R`` and by definition ``\epsilon^2=0`` (instead
of ``i^2=-1`` in complex numbers). What are the properties of these Dual numbers?
```math
(v + \dot v \epsilon) + (u + \dot u \epsilon) = (v + u) + (\dot v + \dot u)\epsilon  \\
(v + \dot v \epsilon)(u + \dot u \epsilon) = vu + (u\dot v + \dot u v)\epsilon + \dot v \dot u \epsilon^2 = vu + (u\dot v + \dot u v)\epsilon \\
\frac{v + \dot v \epsilon}{u + \dot u \epsilon} = \frac{v + \dot v \epsilon}{u + \dot u \epsilon} \frac{u - \dot u \epsilon}{u - \dot u \epsilon} = \frac{v}{u} - \frac{(\dot u v - u \dot v)\epsilon}{u^2}
```

#### How are dual numbers related to differentiation?
Let's evaluate the above equations at ``(v, \dot v) = (v, 1)`` and ``(u, \dot u) = (u, 0)``
we obtain 
```math
(v + \dot v \epsilon) + (u + \dot u \epsilon) = (v + u) + 1\epsilon  \\
(v + \dot v \epsilon)(u + \dot u \epsilon) = vu + u\epsilon\\
\frac{v + \dot v \epsilon}{u + \dot u \epsilon} = \frac{v}{u}  + \frac{1}{u} \epsilon
```
and notice that terms ``(1, u, \frac{1}{u})`` corresponds to gradient of functions ``(u+v, uv, \frac{u}{v})`` with respect to ``v``. We can repeat it with echanged values of ``\epsilon`` as ``(v, \dot v) = (v, 0)`` and ``(u, \dot u) = (u, 1)``
and we obtain
```math
(v + \dot v \epsilon) + (u + \dot u \epsilon) = (v + u) + 1\epsilon  \\
(v + \dot v \epsilon)(u + \dot u \epsilon) = vu + v\epsilon\\
\frac{v + \dot v \epsilon}{u + \dot u \epsilon} = \frac{v}{u}  - \frac{v}{u^2} \epsilon
```
meaning that at this moment we have obtained gradients with respect to ``u``.

All above functions ``(u+v, uv, \frac{u}{v})`` are of ``\mathbb{R}^2 \rightarrow \mathbb{R}``, therefore we had to repeat the calculations twice to get gradients with respect to both inputs. This is inline with the above theory, where we have said that if input dimension is larger then output dimension, the backward mode is better. But consider a case, where we have a function 
```math
f(v) = (v + 5, 5*v, 5 / v) 
```
which is ``\mathbb{R} \rightarrow \mathbb{R}^3``. In this case, we obtain the Jacobian ``[1, 5, -\frac{5}{v^2}]`` in a single forward pass (whereas the reverse would require three passes over the backward calculation, as will be seen later).

#### Does the above holds universally?
Let's first work out polynomial. Let's assume the polynomial
```math
p(v) = \sum_{i=1}^n p_iv^i
```
and compute its value at ``v + \dot v \epsilon`` (note that we know how to do addition and multiplication)
```math
p(v) = \sum_{i=0}^n p_i(v + v \epsilon )^i = \sum_{i=0}^n \left[p_i \sum_{j=0}^{n}\binom{i}{j}v^{i-j}(\dot v \epsilon)^{i}\right] = p_0 + \sum_{i=1}^n \left[p_i \sum_{j=0}^{1}\binom{i}{j}v^{i-j}(\dot v \epsilon)^{j}\right] = p_0 + \sum_{i=1}^n p_i(v^i + i v^{i-1} \dot v \epsilon ) = p(v) + \left(\sum_{i=1}^n ip_i v^{i-1}\right) \dot v \epsilon
```
where in the multiplier of ``\dot v \epsilon``, ``\sum_{i=1}^n ip_i v^{i - 1}\right``, we recognize the derivative of `p(v)` with respect to `v`. This proves that Dual numbers can be used to calculate the gradient of polynomials.

Let's now consider a general function ``f:\mathbb{R} \rightarrow \mathbb{R}``. Its value at point ``v + \dot v \epsilon`` can be approximated using Taylor expansion at function at point `v` as
```math
f(v+\dot v \epsilon) = \sum_{i=0}^\infty \frac{f^i(v)\dot v^i\epsilon^n}{i!}
  = f(v) + f'(v)\dot v\epsilon,
```
where all higher order terms can be dropped because ``\epsilon^i=0`` for ``i>1``. This shows that we can calculate the gradient of ``f`` at point `v` by calculating its value at `f(v + \epsilon)` and taking the multiplier of `\epsilon`.

#### Implementing Dual number with Julia
To demonstrate the simplicity of Dual numbers, consider following definition of Dual numbers, where we define a new number type and overlad functions `+`, `-`, `*`, and `/`.  In Julia, this reads:
```@example lec08
struct Dual{T<:Number} <: Number
    x::T
    d::T
end

Base.:+(a::Dual, b::Dual)   = Dual(a.x+b.x, a.d+b.d)
Base.:-(a::Dual, b::Dual)   = Dual(a.x-b.x, a.d-b.d)
Base.:/(a::Dual, b::Dual)   = Dual(a.x/b.x, (a.d*b.x - a.x*b.d)/b.x^2) # recall  (a/b) =  a/b + (a'b - ab')/b^2 ϵ
Base.:*(a::Dual, b::Dual)   = Dual(a.x*b.x, a.d*b.x + a.x*b.d)

# Let's define some promotion rules
Dual(x::S, d::T) where {S<:Number, T<:Number} = Dual{promote_type(S, T)}(x, d)
Dual(x::Number) = Dual(x, zero(typeof(x)))
Dual{T}(x::Number) where {T} = Dual(T(x), zero(T))
Base.promote_rule(::Type{Dual{T}}, ::Type{S}) where {T<:Number,S<:Number} = Dual{promote_type(T,S)}
Base.promote_rule(::Type{Dual{T}}, ::Type{Dual{S}}) where {T<:Number,S<:Number} = Dual{promote_type(T,S)}

# and define api for forward differentionation
forward_diff(f::Function, x::Number) = _dual(f(Dual(x,1.0)))
_dual(x::Dual) = x.d
_dual(x::Vector) = _dual.(x)
```

And let's test the **_Babylonian Square Root_** (an algorithm to compute $\sqrt x$):
```@repl lec08
babysqrt(x, t=(1+x)/2, n=10) = n==0 ? t : babysqrt(x, (t+x/t)/2, n-1)

forward_diff(babysqrt, 2) 
forward_diff(x -> [1 + x, 5x, 5/x], 2) 
forward_diff(babysqrt, 2) ≈ 1/(2sqrt(2))
```

We now compare the analytic solution to values computed by the `forward_diff`
```math
f(x) = \sqrt{x} \qquad f'(x) = \frac{1}{2\sqrt{x}}
```
```@repl lec08
forward_dsqrt(x) = forward_diff(babysqrt,x)
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
2. To make the forward diff work in julia, we only need to **_overload_** a few **_operators_** for forward mode AD to
   work on **_any function_**
3. For vector valued function we can use **_Hyperduals_**
5. Forward diff can differentiation through the `setindex!` (more on this later on)
6. ForwardDiff is implemented in `ForwardDiff.jl`, which might appear to be neglected, but the truth is that it is very stable and general implementation.
7. ForwardDiff does not have to be implemented through Dual numbers. It can be implemented similarly to ReverseDiff through multiplication of Jacobians, which is what is the community work on now (in `Diffractor`, `Zygote` with rules defined in `ChainRules`).
---

## Reverse mode
In reverse mode, the computation of the gradient follow the oposite order. That means.
We initialize the Jacobian by ``\frac{\partial y}{\partial y_0},`` which is again an identity matrix, and perform the computation of jacobians and multiplications in the opposite order. The problem is that to calculate jacobian ``J_i``, we need to know ``y_i``. Therefore the reverse mode diff requires two passes over the computation graph.

1. Forward pass: iterate `i` from `n` down to `1` as
    - calculate the next intermediate output as ``y_{i-1} = f_i(y_i)`` 

2. Backward pass: iterate `i` from `1` down to `n` as
    - calculate Jacobian ``J_i = \frac{f_i}{\partial y_i}`` at point ``y_i``
    - *pull back* the gradient as ``\frac{\partial y_0}{\partial y_{i}} = \frac{\partial y_0}{\partial y_{i-1}} \times J_i``
    Notice that we actually cannot compute the backward pass, as we do not know ``y_i`` at the moment of computing the `J_i`. Therefore the backward pass needs to be preceeded by forward pass, where we calculate ``y_i`` as follows

The fact that we need to store intermediate outs has a huge impact on the memory requirements. Therefore when we have been talking few lectures ago that we should avoid excessive memory allocations, here we have an algorithm where the excessive allocation is by design. 

Explain complications of the `setindex!`


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

## Graoh-based AD

!(diff graph)[graphdiff.jl]

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


* [simplest viable implementation](https://juliadiff.org/ChainRulesCore.jl/dev/autodiff/operator_overloading.html#ReverseDiffZero)

## What are the tricks that we can use?
- we define custom rules over large functional blocks. For example while we can auto-grad (in theory) matrix product, it is much more efficient to define make a matrix multiplication as one large function, for which we define jacobians (not to say that by doing so, we can dispatch on Blas)
- **Invertible functions** When we are differentiating invertible functions, we can calculate intermediate outputs from the 
- **Checkpointing** We can store the intermediate ouputs only sometime and performed a small forward while doint backward
- ** Implicit functions** Differentiating through solvers is possible, since they are mostly iterative. But, using the mathematical equality is easier.

!!! info
    Reverse mode AD was first published in 1976 by Seppo Linnainmaa, a finnish computer scientist. It was popularized in the end of 80s when applied to training multi-layer perceptrons, which gave rise to the famous **backpropagation** algorithm, which is a special case of reverse mode AD.
    *Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986), Learning representations by back-propagating errors., Nature, 323, 533--536.*

### Implementation
- TensorFlow with explicit graph vs. PyTorch with tape / Wengert list

### How to test
A tricky example with odd function

## Why custom rules
- We need them for speed
- They are needed for numerical stability

### How to test
A tricky example with odd function

### ChainRules
- Why you want to have it
- Syntax
- Structural vs Natural gradient

## Chain Rules & Jacobians

# Sources

* Mike Innes' [diff-zoo](https://github.com/MikeInnes/diff-zoo)
* [Write Your Own StS in One Day](https://blog.rogerluo.me/2019/07/27/yassad/)
* [Zygote.jl Paper](https://arxiv.org/pdf/1810.07951.pdf)
  and [Zygote.jl Internals](https://fluxml.ai/Zygote.jl/dev/internals/)
* Keno's [Talk](https://www.youtube.com/watch?v=mQnSRfseu0c&feature=youtu.be)
* Chris' [Lecture](https://mitmath.github.io/18337/lecture11/adjoints)
