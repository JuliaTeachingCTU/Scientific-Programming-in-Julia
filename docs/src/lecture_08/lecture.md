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
