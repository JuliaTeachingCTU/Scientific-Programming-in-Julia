using KernelAbstractions

# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(a, b, c)
    i, j = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(c))
    for k = 1:size(a)[2]
        tmp_sum += a[i,k] * b[k, j]
    end

    c[i,j] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function matmul!(a, b, c)
    if size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    backend = KernelAbstractions.get_backend(a)
    kernel! = matmul_kernel!(backend)
    kernel!(a, b, c, ndrange=size(c))
end

using Metal
a = rand(Float32, 10000, 10000)
b = rand(Float32, 10000, 10000)
ag = a |> MtlArray
bg = b |> MtlArray
c = similar(ag)
matmul!(ag,bg,c)

@assert a*b ≈ Matrix(c)

@btime $a*$b
@btime Metal.@sync matmul!($ag, $bg, $c)
# KernelAbstractions.synchronize(backend)

# @test isapprox(c, a*b)
