using CUDA

# define a dense layer
struct Dense{W<:AbstractArray,B<:AbstractArray,F}
	w::W 
	b::B
	f::F
end 

function Dense(idim, odim, f = identity)
	Dense(randn(Float32, odim, idim), randn(Float32, odim), f)
end

function (l::Dense)(x)
	l.f.(l.w * x .+ l.b)
end

#define moving of data to CPU
gpu(x::AbstractArray) = CuArray(x)
cpu(x::CuArray) = Array(x)
gpu(l::Dense) = Dense(gpu(l.w), gpu(l.b), l.f)
gpu(l::ComposedFunction) = gpu(l.outer) ∘ gpu(l.inner)

# a simple but powerful non-linearity
relu(x::T) where {T<:Number} = max(x, zero(T))


# Let's now define a small one hidden layer neural network
x = randn(Float32, 16, 100)
l₁ = Dense(16,32, relu)
l₂ = Dense(32,8)
nn = l₂ ∘ l₁

# and try to profile a computation
CUDA.@profile CUDA.@sync begin 
    NVTX.@range "moving nn to gpu" gpu_nn = gpu(nn)
    NVTX.@range "moving x to gpu" gpu_x = gpu(x)
    NVTX.@range "nn(x)" o = gpu_nn(gpu_x)
    NVTX.@range "moving results to cpu" cpu(o)
end
