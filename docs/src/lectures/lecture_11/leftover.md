
```julia
cx = CUDA.rand(1024,1024)
cb = CUDA.zeros(1)
@macroexpand  @cuda threads=128 blocks=8192 reduce_warp(+, cx, cb, 0f0)

quote
    var"##415" = (+)
    var"##416" = CUDA.rand(1024, 1024)
    var"##417" = CUDA.zeros(1)
    var"##418" = 0.0f0
    begin
        var"##f_var#419" = reduce_warp
        $(Expr(:gc_preserve, quote
    	local var"##kernel_f#420" = (CUDA.cudaconvert)(var"##f_var#419")
    	local var"##kernel_args#421" = map(CUDA.cudaconvert, (var"##415", var"##416", var"##417", var"##418"))
    	local var"##kernel_tt#422" = Tuple{map(Core.Typeof, var"##kernel_args#421")...}
    	local var"##kernel#423" = (CUDA.cufunction)(var"##kernel_f#420", var"##kernel_tt#422"; )
	    if true
	        var"##kernel#423"(var"##415", var"##416", var"##417", var"##418"; $(Expr(:(=), :threads, 128)), $(Expr(:(=), :blocks, 8192)))
	    end
	    var"##kernel#423"
	end, Symbol("##415"), Symbol("##416"), Symbol("##417"), Symbol("##418"), Symbol("##f_var#419")))
    end
end
```
prepares arguments to compile the kernel, which is done in the function `CUDA.cufunction` and we can prepare it as 

```julia
f = CUDA.cudaconvert(reduce_warp)
cuparams = CUDA.cudaconvert((+, cx, cb, 0f0))
tt = Tuple{map(Core.Typeof, cuparams)...}
kernel_struct = CUDA.cufunction(f, tt)
```

Diving into the `CUDA.cufunction` at `CUDA.jl/src/compiler/execution.jl:290` we observe that it prepares the compilation job and send it to the GPU compiler, which either compiles it or fetch it from the the cache.
```julia
function cufunction(f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    cuda = active_state()
    cache = cufunction_cache(cuda.context)
    source = FunctionSpec(f, tt, true, name)
    target = CUDACompilerTarget(cuda.device; kwargs...)
    params = CUDACompilerParams()
    job = CompilerJob(target, source, params)
    return GPUCompiler.cached_compilation(cache, job,
                                          cufunction_compile,
                                          cufunction_link)::HostKernel{F,tt}
end
```

Let's now dive into the compilation part. We get a sense using `Cthulhu.jl`, but sometimes, it is good to do this by hand
```julia 
using CUDA: active_state, cufunction_cache, FunctionSpec, CUDACompilerTarget, CUDACompilerParams, cufunction_compile, cufunction_link
cuda = active_state()
cache = cufunction_cache(cuda.context)
source = FunctionSpec(f, tt, true, nothing)
target = CUDACompilerTarget(cuda.device)
params = CUDACompilerParams()
job = CompilerJob(target, source, params)
GPUCompiler.cached_compilation(cache, job, cufunction_compile, cufunction_link)
```
* `FunctionSpec` is just a struct contains information about a function that will be compiled
* `CompilerJob` is a structure containing all important information for compilation of the kernel
* `GPUGCompile.cached_compilation` is a cache, which either fetches the kernel from the cache, or force the compilation. Let's look at the compilation

Lowerint to ptx takes form in 
```julia
# lower to PTX
mi, mi_meta = GPUCompiler.emit_julia(job)
ir, ir_meta = GPUCompiler.emit_llvm(job, mi)
asm, asm_meta = GPUCompiler.emit_asm(job, ir; 
	format=LLVM.API.LLVMAssemblyFile)
```
