# [Homework 9: Something parallel](@id hw09)

## How to submit
Put all the code of inside `hw.jl`. Zip only this file (not its parent folder) and upload it to BRUTE. You should not not import anything but `Base.Threads` or just `Threads`. 

```@raw html
<div class="admonition is-category-homework">
<header class="admonition-header">Homework (2 points)</header>
<div class="admonition-body">
```
Implement *multithreaded* discrete 1D convolution operator[^1] without padding (output will be shorter). The required function signature: `thread_conv1d(x, w)`, where `x` is the signal array and `w` the kernel. For testing correctness of the implementation you can use the following example of a step function and it's derivative realized by kernel `[-1, 1]`:
```julia
using Test
@test all(thread_conv1d(vcat([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]), [-1.0, 1.0]) .≈ [0.0, -1.0, 0.0, 1.0, 0.0])
```
[^1]: Discrete convolution with finite support [https://en.wikipedia.org/wiki/Convolution#Discrete\_convolution](https://en.wikipedia.org/wiki/Convolution#Discrete_convolution)

Your parallel implementation will be tested both in sequential and two threaded mode with the following inputs
```julia
using Random
Random.seed!(42)
x = rand(10_000_000)
w = [1.0, 2.0, 4.0, 2.0, 1.0]
@btime thread_conv1d($x, $w);
```
On your local machine you should be able to achieve `0.6x` reduction in execution time with two threads, however the automatic eval system is a noisy environment and therefore we require only `0.8x` reduction therein. This being said, please reach out to us, if you encounter any issues.

**HINTS**:
- start with single threaded implementation
- don't forget to reverse the kernel
- `@threads` macro should be all you need
- for testing purposes create a simple script, that you can run with `julia -t 1` and `julia -t 2`

```@raw html
</div></div>
<details class = "solution-body" hidden>
<summary class = "solution-header">Solution:</summary><p>
```

Nothing to see here

```@raw html
</p></details>
```