# [Lab 11: GPU programming](@id gpu_lab)

!!! warning "Disclaimer"
    With the increasing complexity of computer HW some statements may become outdated. Moreover we won't cover as many tips that you may encounter on a gpu parallel programming specific course, which will teach you more in the direction of how to think in parallel, whereas here we will focus on the tools that you can use to realize the knowledge gained therein. Lastly even though we've used NVidia gpus, the same can be applied to AMD gpus, which offers the open source CUDA alternatives HIP/ROCM. *TODO LINK* We have chosen to use primarily NVidia CUDA for demonstration due to it's maturity and the availability of HW on our side.


## We don't want to get our hands dirty
We can do quite a lot without even knowing that we are using GPU instead of CPU. This marvel is the combination of Julia's multiple dispatch and array abstractions. Based on the size of the problem and intricacy of the computation we may be achieve both incredible speedups as well as slowdowns. 

### Something linear
### Image processing
### Some linear algebra
### Caveats, remarks and the holy grail (Flux, Tulio)
Scalar indexing - even just printing
Deeper for loop acceleration
The opaqueness of machine learning with Flux


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