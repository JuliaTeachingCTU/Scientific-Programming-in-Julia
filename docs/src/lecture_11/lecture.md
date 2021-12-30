# GPU programming
## How GPU differs from CPU
### Hardware perspective
**CPU** was originally created for maximal throughput of a single threadded program. Therefore the modern CPU has many parts which are not devoted to the actual computation, but to maximize the utilization of a computing resource (ALU), which is relatively small. Below is the picture of a processor of Intel's Core architecture (one of the earliest in the series).
![cpu](Intel_Core2.png)
![cpu die](skylake_core_die.png)

It contains blocks allowing to execute instructions in parallel, out of order, speculatively just to maximally utilize all computational units.
Notable functionalities / blocks are
* **superscalar execution** is an ability to execute more than one instruction at the given time.

* **Reoreder buffer** reorders the execution of instructions such that two non-interfering instructions can be executed simultaneously

* **Register renaming** renames registers, such that two non-interfering instructions over the same register can be executed together

* **Branch prediction** predicts which branch will be taken and execute instructions in that branch. Misses are costly, as the processor needs to roll back emptying the instruction pipeline. The processor state is nor fully restored, which leads to side-channel attacks.

* **speculative prefetching** load instruction / data from memory to caches in advance in the hope they will be executed (depends on branch predictions)

* **Memory management unit** is not shown but takes care of translation of virtual addresses to physical, checking the security bits of pages, etc.

* **Caches** (three levels) thrive to provide instuctions with data from cache, such that it does not have to wait for the load. Caches are transparent to the user, he does not control, what will stay in cache.

* **L1 Cache synchronization** If the processor contains many cores, their L1 cache is atomically synchronized.

* **Buffers** are used to cache for example mapping of virtual to physical addresses, partial computations to allow rollbacks.

* **Interrupt management** CPU can interrupt its execution and transfer the execution to a different location, changing security levels.

**GPU** was from the very beginning designed for maximal throughput achieved by parallelism. The reason for this is simple. Imagine that you need to render a 4K image (resolution 3840 × 2160 = 8 294 400 pixels) with refresh rate 60fps in the first person shooter game. This means that you need to compute intensities of 0.5G pixels per second. But, the program computing the intensity of a pixel is the same, as what you do is a something like
```julia
for (i,j) in Itertors.Product(1:2160, 1:3840)
	image[i,j] = compute_insity(i, j)
end
```
and the computation of intensities `compute_insity(i, j)` does not contain many branches. Therefore the GPU goes for massive parallelism while simplifying each Core the the bare minimum leaving all difficulties up to the programmer / compiler. Below we see a high-level view on Nvidia's GPU.
![nvidia-gpu](nvidia-gpu.jpg)
![nvidia-gpu](nvidia-kepler.jpg)
1. The chip contains many streaming multi-processors (SM). Normally, each streamming processor would be called core, as it is an indivisible unit, but NVidia decided to call "a core" something which in normal CPU would be called arithmetic unit (which certainly helps the marketting).
2. Each streaming multi-processor contains (possibly multiple blocks of) 32-way (32-wide) SIMT units (Nvidia calls that *CUDA Core*), shared memory (managed cache) and register file (16k registers, but shared between all threads). Therefore a pascal P-100 can have up to 64×4×32 = 8192 cores, which certainly sounds cool in comparison to for example 24 cores of normal processors, but.
3. Each streaming multi-processors (SM) has one instruction fetch and decode unit, which means that *all* CUDA cores of that SM *has to* execute the same instruction at a given cycle. This simplifies the design. The execution model therefore roughly corresponds to vector (SIMD) registers in normal CPU, but CUDA cores are not as restricted as SIMD registers. NVidia therefore calls this computation model single instruction mulptiple threads (SIMT). Main differences:
    1. SIMD requires the memory to be continuous while SIMT does not. 
    2. SIMT is explicitly scalar while SIMD is explicitly vector. 
32 CUDA cores each operating over 32 bit registers would be equal to 1024 long vector (SIMD) registers. Modern Intel XEON processors has 256 / 512 long registers, which seems similar, as said above in order to use them the data has to be aligned in memory and sometimes they has to be on a particular offset, which might be difficult to achieve in practice (but to be fair, if the data are not aligned in GPU, the loading of data is very inefficcient).
4. 16k registers per SM might seem like a lot, but they are shared between all threads. In modern GPUs, each SM supports up to 2048 threads, which means there might be just 4 32-bit registers per thread.
5. GPUs do not have virtual memory, interrupts, and cannot address external devices like keyboard and mouse.
6. GPUs can switch execution contexts of "set of threads" at no cost. In comparison the context switch in CPU is relatively expensive (we need to at least save the content of registers, which is usually sped by having two sets of registers). This helps to hide latencies, as when a set of threads is stalled, other set of threads are available.
7. The programmer deal with "raw" storage hierarchy. This means that the programmer has to manage what will be and what will not be in cache by itself, on GPU, they are exposed and you decide what will be loaded into the cache. 
8. Caches are synchronized only within a single SM, this is unlike in CPU, where L1 caches are synchronized across cores, according to some leading in bottleneck in having more cores.
9. SM has relatively low frequency clocks, which helps to deal with thermal problems.
10. The memory in SM is divided into 16 banks, write operations to a bank is sequential. If two threads are writing to the same bank, they can be stalled.


### Programming / Execution model
The GPU works in an asynchronous mode. If we want to execute something on GPU, we need to 
1. Upload the data to GPU memory (if they are not already there)
2. Compile the *kernel* --- a code to be executed on GPU (if not already compiled)
3. Upload the kernel code to GPU
4. Request the computation of the kernel (more on this below). GPU will put the request for the computation to its queue of works to be performed and once resources are free, it will do the compuation.
5. The control is immediately returned, which means that CPU can continue doing its job. 
6. CPU can issue a blocking wait till the computation is done, for example fetching the results.

Let's now look at point 4. in more detail.
Recall that GPU is designed for processing data in parallel, something we can abstract as 
```julia
for i in 1:N
	kernel!(i)
end
``` 
where `kernel!(i)` means compute the `i`-th item of the data using function `kernel!`, which is modifying as it does not have a return value, but has to put all results to the preallocated array. `i`-th part of the data usually corresponds to one float number (usually `Float32`). Here, we can see that SIMT has a scalar notation, we refer to individual numbers inside arrays.

Each item, `kernel!(i)`, is executed on a single *thread*. GPU *always* execute 32 threads at once on a single SM. This group of 32 threads is called **warp**. These 32 threads within warp can very effectively communicate with each other using atomic instructions. A user has to group threads into **group**. Each group is executed on a single SM, which means that all threads within this group has access to fast SM local memory. All blocks of single job are called **grid**.
* From the above we can already see that the number of threads has to be multiple of 32 (given by the requirement on warps).
* Each block can have up to 2048 threads, which are executed on a single SM. Large number of threads in a single block is good if 
    * those threads needs to access the same memory (for example in Stencil operations), which can be put to the local cache and 
    * each thread reads data from memory (which are no co-allesced) and the SM can run different thread while other is stalling (mostly due to waiting for finishing loading data from memory). 
On the other hand large number of threads per group might stall due to insufficient number of registers and / or other threads being busy.
* The total number of issued threads  has to be multiple of 32 and of the number of threads per block, hence *there will always be threads that does not do anything*.
* The division of a total number of items to be processed `N` into blocks is therefore part of the problem and it can be specific to a version of GPU.
* For some operations (e.g. reduction seen below) to get the highest performance, you need to write the same algorithm for three levels of sets of threads  --- warp, groups, and grid. This is the price paid for exposed cache levels (we will se an example below on reduction).

As has been mentioned above, all CUDA cores in one SM are processing the same instruction. therefore if the processed kernel contains conditional statements `if / else` blocks, both blocks will be processed in sequential order as illustrated below,
![simt](simt-divergence.png)

which can significantly decrease the throughput.

**Latency hiding** 
A thread can stall, because the instruction it depends on has not finished yet, for example due to loading data from memory, which can be very time consuming (recall that unlike SIMD, SIMT can read data from non-coallesced memory location at the price of increased latency). Instead of waiting, SM will switch to execute different set of threads, which can be executed. This context switch is does not incur any overhead, hence it can occur at single instruction granularity. It keeps SM busy effective hiding latency of expensive operations.
![latency hiding](latency-hiding.jpg)
[image taken from](https://iq.opengenus.org/key-ideas-that-makes-graphics-processing-unit-gpu-so-fast/)

## using GPU without writing kernels
Julia allows to execute things on GPU as you would do on CPU. Thanks to Julia's multiple dispatch, it is sufficient to conver the `Array` to `CuArray` and the magic happens.

```julia
using CUDA
using BenchmarkTools

x = randn(Float32, 1000, 1000)
y = randn(Float32, 1000, 1000)
x * y 
cx = CuArray(x)
cy = CuArray(y)
cx * cy
x * y ≈ Matrix(cx * cy)
julia> @btime x * y;
  5.737 ms (2 allocations: 3.81 MiB)

julia> @btime cx * cy;
  18.690 μs (32 allocations: 624 bytes)

julia> @btime CUDA.@sync cx * cy;
  173.704 μs (32 allocations: 624 bytes)
```
The see that matrix multiplication on CUDA is about 33x faster, which is likely caused by being optimized by directly NVidia, as `cx * cy` calls `CuBlas` library.

How much does it cost to send the matrix to GPU's memory? Let's measure the time of the roundtrip
```julia
julia> @btime Matrix(CuMatrix(x));
  1.059 ms (9 allocations: 3.81 MiB)
```

`CUDA.jl` implements some function that allows to perform some non-trivial operation, such as generic `map` and `reduce` (albeit as we will se later, performant `reduce` operation is very difficult).
```julia
sin.(cx).^2 .+ cos.(cx).^2
map(x -> sin(x)^2 + cos(x)^2, cx)
reduce(max, cx)
reduce(max, cx, dims = 1)
```
which helps in many situations. Julia here benefits from JAoT compilation, as it can compile kernels on the fly (though you might not get the highest performance).

Let's now try to use CUDA on computation of Julia set, which should benefit a lot from CUDA's paralelization, as we can dispatch each pixel to each thread --- something GPUs were originally designed for. 

We slightly modify the kernel we have used in our lecture on multi-threadding, mainly to force all types to be 32-bit wide
```julia
using CUDA
using BenchmarkTools

function juliaset_pixel(i, j, n, c)
    z = ComplexF32(-2f0 + (j-1)*4f0/(n-1), -2f0 + (i-1)*4f0/(n-1))
	for i in UnitRange{Int32}(0:255)
        abs2(z)> 4.0 && return(i)
        z = z*z + c
	end
    return(i)
end

c = ComplexF32(-0.79f0, 0.15f0);
n = Int32(1000);
is = collect(Int32, 1:n)';
js = collect(Int32, 1:n);
img = juliaset_pixel.(is, js, n, c);

cis = CuArray(is);
cjs = CuArray(js);
img = juliaset_pixel.(cis, cjs, n, c);
@btime juliaset_pixel.(cis, cjs, n, c);
@btime CUDA.@sync juliaset_pixel.(cis, cjs, n, c);
```
We see that CPU version is much faster then the GPU one? Why is that? My suspect is that the kernel needs to execute all 255 iterations of the while for each pixel, which is very wasteful, especially considering that many / most pixels require one iteration. Hence, we see the effect of thread divergence in practice (recall that all threads within the block has to execute all instructions, even though some of them are masked.)

!!! info 
	### Profiler
    Cuda offers a sampling profiler. It is available in `nsight-systems` and you have to download it separately and launch julia within the profiler as for example `/opt/nvidia/nsight-systems/2021.5.1/bin/nsys launch --trace=cuda /home/pevnak/julia-1.7.1/bin/julia  --color=yes`.
    and then, we can profile the code using the usual `@profile` macro this time sourced from `CUDA` as
    ```julia
    CUDA.@profile CUDA.@sync juliaset_pixel.(cis, cjs, n);
    ```
    the report is saved to `report???.???` (nvidia likes to change the suffix) and it can be inspected by `nsys-ui` interactive tool. **Do not forget to run the profiler twice to get rid of compilation artifacts.**
    You can further anotate parts of your code as 
    ```
    CUDA.@profile CUDA.@sync begin 
	    NVTX.@range "julia set" juliaset_pixel.(cis, cjs, n);
    end
    ```
    for better orientation in the code.

In the output of the profiler we see that there is a lot of overhead caused by launching the kernel itself and then, the execution is relatively fast. 


While Julia's JAoT greatly enhances the power of prepared kernels, you might quickly run into a case, when you are able to perform the operation on GPU, but it is very slow. Sometimes, it might be just faster to move the array to CPU, perform the operation there and move it back to GPU. Although this sounds like a pretty bad idea, it actually works very well see below.
```julia
using Mill
using Random
using CUDA
using BenchmarkTools
n = vcat(rand(1:10,1000), rand(11:100, 100), rand(101:1000,10))
x = randn(Float32, 128, sum(n))
z = zeros(Float32, 128, 1)
bags = Mill.length2bags(n)

builtin(x, bags, z) = Mill.segmented_sum_forw(x, vec(z), bags, nothing)

function naive(x, bags, z)
	o = similar(x, size(x,1), length(bags))
	foreach(enumerate(bags)) do (i,b)
		if isempty(b)
			o[:,i] .= z
		else
			@inbounds o[:,i] = sum(@view(x[:,b]), dims = 2)
		end
	end
	o
end

builtin(x, bags, z) ≈ naive(x, bags, z)
@btime builtin(x, bags, z);
@btime naive(x, bags, z);

cx = CuArray(x);
cz = CuArray(z);
naive(cx, bags, cz);
@btime CUDA.@sync naive(cx, bags, cz);
@btime CUDA.@sync CuArray(builtin(Array(cx), bags, Array(cz)));
```

http://mikeinnes.github.io/2017/08/24/cudanative.html
## Writing CUDA kernels

## How Julia compiles CUDA kernels


## Sources
* [SIMD < SIMT < SMT: parallelism in NVIDIA GPUs](http://yosefk.com/blog/simd-simt-smt-parallelism-in-nvidia-gpus.html)