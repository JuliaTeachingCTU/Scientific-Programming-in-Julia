# [Lab 11: GPU programming](@id gpu_lab)
In this lab we are going to delve into the topic of using GPU hardware in order to accelerate scientific computation. We will focus mainly on NVidia graphics hardware and it's [CUDA](https://developer.nvidia.com/cuda-zone) framework[^1], however we will no go into as much detail and thus what you learn today should be aplicable to alternative HW/frameworks such as AMD's [HIP/ROCM](https://www.amd.com/en/graphics/servers-solutions-rocm) or Intel's [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html#gs.lfd9th). We have chosen to use primarily NVidia CUDA for demonstration due to it's maturity and the availability of HW on our side.

!!! warning "Disclaimer"
    With the increasing complexity of GPU HW some statements may become outdated. Moreover we won't cover as many tips that you may encounter on a GPU parallel programming specific course.

## We DON'T want to get our hands dirty
We can do quite a lot without even knowing that we are using GPU instead of CPU. This marvel is the combination of Julia's multiple dispatch and array abstractions. Based on the size of the problem and intricacy of the computation we may be achieve both incredible speedups as well as slowdowns. 

The gateway to working with CUDA in Julia is the `CUDA.jl` library, which offers the following user facing functionalities
- device management `versioninfo`, `device!`
- definition of arrays on gpu `CuArray`
- data copying from host(CPU) to device(GPU) and the other way around
- wrapping already existing library code in `CuBLAS`, `CuRAND`, `CuDNN`, `CuSparse` and others
- kernel based programming (more on this in the second half of the lab)

Let's use this to inspect our GPU hardware.
```julia
julia> using CUDA
julia> CUDA.versioninfo()
CUDA toolkit 11.5, artifact installation
NVIDIA driver 460.56.0, for CUDA 11.2
CUDA driver 11.2

Libraries: 
- CUBLAS: 11.7.4
- CURAND: 10.2.7
- CUFFT: 10.6.0
- CUSOLVER: 11.3.2
- CUSPARSE: 11.7.0
- CUPTI: 16.0.0
- NVML: 11.0.0+460.56
- CUDNN: 8.30.1 (for CUDA 11.5.0)
- CUTENSOR: 1.3.3 (for CUDA 11.4.0)

Toolchain:
- Julia: 1.6.5
- LLVM: 11.0.1
- PTX ISA support: 3.2, 4.0, 4.1, 4.2, 4.3, 5.0, 6.0, 6.1, 6.3, 6.4, 6.5, 7.0
- Device capability support: sm_35, sm_37, sm_50, sm_52, sm_53, sm_60, sm_61, sm_62, sm_70, sm_72, sm_75, sm_80

2 devices:
  0: GeForce RTX 2080 Ti (sm_75, 10.111 GiB / 10.758 GiB available)
  1: TITAN Xp (sm_61, 11.897 GiB / 11.910 GiB available)
```

!!! info "CUDA.jl compatibility"
	The latest development version of CUDA.jl requires Julia 1.6 or higher. `CUDA.jl` currently also requires a CUDA-capable GPU with compute capability 3.5 (Kepler) or higher, and an accompanying NVIDIA driver with support for CUDA 10.1 or newer. These requirements are not enforced by the Julia package manager when installing CUDA.jl. Depending on your system and GPU, you may need to install an older version of the package.[^1]

	[^1]: Disclaimer on `CUDA.jl`'s GitHub page: [url](https://github.com/JuliaGPU/CUDA.jl)

As we have already seen in the lecture *TODO LINK*, we can simply import `CUDA.jl` define some arrays, move them to the GPU and do some computation. In the following code we define two matrices `1000x1000` filled with random numbers and multiply them using usuall `x * y` syntax.
```julia
x = randn(Float32, 60, 60)
y = randn(Float32, 60, 60)
x * y 
cx = CuArray(x)
cy = CuArray(y)
cx * cy
x * y ≈ Matrix(cx * cy)
```
This may not be anything remarkable, as such functionality is available in many other languages albeit usually with a less mathematical notation like `x.dot(y)`. With Julia's multiple dispatch, we can simply dispatch the multiplication operator/function `*` to a specific method that works on `CuArray` type. Check with `@code_typed` shows the call to CUBLAS library under the hood.
```julia
julia> @code_typed cx * cy
CodeInfo(
1 ─ %1 = Base.getfield(A, :dims)::Tuple{Int64, Int64}
│   %2 = Base.getfield(%1, 1, true)::Int64
│   %3 = Base.getfield(B, :dims)::Tuple{Int64, Int64}
│   %4 = Base.getfield(%3, 2, true)::Int64
│   %5 = Core.tuple(%2, %4)::Tuple{Int64, Int64}
│   %6 = invoke CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}(CUDA.undef::UndefInitializer, %5::Tuple{Int64, Int64})::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
│   %7 = invoke CUDA.CUBLAS.gemm_dispatch!(%6::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, A::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, B::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, true::Bool, false::Bool)::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
└──      return %7
) => CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
```

Let's now explore what the we can do with this array programming paradigm on few practical examples


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
*we can use test images*
Load a sufficiently large image to the GPU such as this [one]()*TODO LINK* and manipulate it in the following ways:
- create a negative
- half the pixel brightness
- find the brightest pixels
- get it's FFT image

Measure the runtime difference with `BenchmarkTools`. Load the image with the following code, which adds all the necessary dependencies and loads the image into Floa32 matrix.

```julia
using Pkg; 
Pkg.add(["FileIO", "ImageMagick", "ImageShow", "ColorTypes", "FFTW"])

using FileIO, ImageMagick, ImageShow, ColorTypes
rgb_img = FileIO.load("image.jpg");
gray_img = Float32.(Gray.(rgb_img));
cgray_img = CuArray(gray_img);
```

**HINTS**:
- use `Float32` everywhere for better performance
- use `CUDA.@sync` during benchmarking in order to ensure that the computation has completed

**BONUS**: Remove high frequency signal by means of modifying Fourier image.

!!! warning "Scalar indexing"
	Some operations such as showing an image calls fallback implementation which requires `getindex!` called from the CPU. As such it is incredibly slow and should be avoided. In order to show the image use `Array(cimg)` to move it as a whole. Another option is to suppress the output with semicolon
	```julia
	julia> cimg
	┌ Warning: Performing scalar indexing on task Task (runnable) @0x00007f25931b6380.
	│ Invocation of getindex resulted in scalar indexing of a GPU array.
	│ This is typically caused by calling an iterating implementation of a method.
	│ Such implementations *do not* execute on the GPU, but very slowly on the CPU,
	│ and therefore are only permitted from the REPL for prototyping purposes.
	│ If you did intend to index this array, annotate the caller with @allowscalar.
	└ @ GPUArrays ~/.julia/packages/GPUArrays/gkF6S/src/host/indexing.jl:56
	julia> Array(cimg)
	Voila!
	julia> cimg;
	```

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
negative(i) = 1.0f0 .- i
darken(i) = i .* 0.5f0
using CUDA.CUFFT
using FFTW
fourier(i) = fft(i)
brightest(i) = findmax(i)
```

Benchmarking
```julia
julia> @btime CUDA.@sync negative($cgray_img);
  28.990 μs (28 allocations: 1.69 KiB)

julia> @btime negative($gray_img);
  490.301 μs (2 allocations: 4.57 MiB)

julia> @btime CUDA.@sync darken($cgray_img);
  29.170 μs (28 allocations: 1.69 KiB)

julia> @btime darken($gray_img);
  507.921 μs (2 allocations: 4.57 MiB)

julia> @btime CUDA.@sync fourier($cgray_img);
  609.593 μs (80 allocations: 4.34 KiB)

julia> @btime fourier($gray_img);
  45.891 ms (37 allocations: 18.29 MiB)

julia> @btime CUDA.@sync brightest($cgray_img);
  44.840 μs (89 allocations: 5.06 KiB)

julia> @btime brightest($gray_img);
  2.764 ms (0 allocations: 0 bytes)
```


```@raw html
</p></details>
```

In the next example we will try to solve a system of linear equations $Ax=b$, where A is a large (possibly sparse) matrix.
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Benchmark the solving of the following linear system with `N` equations and `N` unknowns. Experiment with increasing `N` to find a value , from which the advantage of sending the matrix to GPU is significant (include the time of sending the data to and from the device). For the sake of this example significant means 2x speedup. At what point the memory requirements are incompatible with your hardware, i.e. exceeding the memory of a GPU?

```julia
α = 10.0f0
β = 10.0f0

function init(N, α, β, r = (0.f0, π/2.0f0))
    dx = (r[2] - r[1]) / N
    A = zeros(Float32, N+2, N+2)
    A[1,1] = 1.0f0
    A[end,end] = 1.0f0
    for i in 2:N+1
        A[i,i-1] = 1.0f0/(dx*dx)
        A[i,i] = -2.0f0/(dx*dx) - 16.0f0
        A[i,i+1] = 1.0f0/(dx*dx)
    end

    b = fill(-8.0f0, N+2)
    b[1] = α
    b[end] = β
    A, b
end

N = 30
A, b = init(N, α, β)
```

**HINTS**:
- use backslash operator `\` to solve the system
- use `CuArray` and `Array` for moving the date to and from device respectively
- use `CUDA.@sync` during benchmarking in order to ensure that the computation has completed

**BONUS 1**: Visualize the solution `x`. What may be the origin of our linear system of equations?
**BONUS 2**: Use sparse matrix `A` to achieve the same thing. Can we exploit the structure of the matrix for more effective solution?
```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
A, b = init(N, α, β)
cA, cb = CuArray(A), CuArray(b)
A
b
A\b
cA\cb

@btime $A \ $b;
@btime CUDA.@sync Array(CuArray($A) \ CuArray($b));
```

**BONUS 1**:
The system comes from a solution of second order ODR with *boundary conditions*.

**BONUS 2**:
The matrix is tridiagonal, therefore we don't have to store all the entries.
```@raw html
</p></details>
```

Programming GPUs in this way is akin to using NumPy, MATLAB and other array based toolkits, which force users not to use for loops. There are attempts to make GPU programming in Julia more powerful without delving deeper into writing of GPU kernels. One of the attempts is [`Tulio.jl`](https://github.com/mcabbott/Tullio.jl), which uses macros to annotate parallel for loops, similar to [`OpenMP`](https://www.openmp.org/)'s `pragma` intrinsics, which can be compiled to GPU as well.

Note also that Julia's `CUDA.jl` is not a tensor compiler. With the exception of broadcast fusion, which is easily transferable to GPUs, there is no optimization between different kernels from the compiler point of view. Furthermore, memory allocations on GPU are handled by Julia's GC, which is single threaded and often not as aggressive, therefore similar application code can have different memory footprints on the GPU.

Nowadays there is a big push towards simplifying programming of GPUs, mainly in the machine learning community, which often requires switching between running on GPU/CPU to be a one click deal. However this may not always yield the required results, because the GPU's computation model is different from the CPU, see lecture *TODO LINK*. This being said Julia's `Flux.jl` framework does offer such capabilities [^2]

```julia
using Flux, CUDA
m = Dense(10,5) |> gpu
x = rand(10) |> gpu
y = m(x)
y |> cpu
```

[^2]: Taken from `Flux.jl` [documentation](https://fluxml.ai/Flux.jl/stable/gpu/#GPU-Support)

## We DO want to get our hands dirty



### Comparison between `CUDA C` and `CUDA.jl`
Compared to CUDA C there is far less bloat, while having the same functionality
```c
#define cudaCall(err) // check return code for error
#define frand() (float)rand() / (float)(RAND_MAX)

__global__ void vadd(const float *a, const float *b, float *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

const int len = 100;
int main() {
	float *a, *b;
	a = new float[len];
	b = new float[len];
	for (int i = 0; i < len; i++) {
		a[i] = frand(); b[i] = frand();
	}
	float *d_a, *d_b, *d_c;
	cudaCall(cudaMalloc(&d_a, len * sizeof(float)));
	cudaCall(cudaMemcpy(d_a, a, len * sizeof(float), cudaMemcpyHostToDevice));
	cudaCall(cudaMalloc(&d_b, len * sizeof(float)));

	cudaCall(cudaMemcpy(d_b, b, len * sizeof(float), cudaMemcpyHostToDevice));
	cudaCall(cudaMalloc(&d_c, len * sizeof(float)));

	vadd<<<1, len>>>(d_a, d_b, d_c);

	float *c = new float[len];
	cudaCall(cudaMemcpy(c, d_c, len * sizeof(float), cudaMemcpyDeviceToHost));
	cudaCall(cudaFree(d_c));
	cudaCall(cudaFree(d_b));
	cudaCall(cudaFree(d_a));

	return 0;
}
```
same code in Julia with `CUDA.jl`
```julia
function vadd(a, b, c)
	i = (blockIdx().x-1) * blockDim().x + threadIdx().x
	c[i] = a[i] + b[i]
	return
end

len = 100
a = rand(Float32, len)
b = rand(Float32, len)
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)
@cuda threads = (1, len) vadd(d_a, d_b, d_c)
c = Array(d_c)
```

### Writing our first kernel
- Continue with examples from lecture
- Go deeper into the indexing
- Show that if we run less threads the result is not correct

### Image processing with kernels


### Profiling


### BONUS: Matrix multiplication


## GPU vendor agnostic code
There is an interesting direction that is allowed with the high level abstraction of Julia - `KernelAbstractions.jl`, which offer an overarching API over CUDA, AMD ROCM and Intel oneAPI frameworks.