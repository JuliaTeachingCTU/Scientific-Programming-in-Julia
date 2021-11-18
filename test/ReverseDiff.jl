using Test
using Scientific_Programming_in_Julia
using FiniteDifferences

import Scientific_Programming_in_Julia.ReverseDiff
import Scientific_Programming_in_Julia.ReverseDiff.σ
import Scientific_Programming_in_Julia.ReverseDiff.data

@testset "ReverseDiff" begin
    # test hcat rule
    x = rand(3,1)
    y = rand(3,1)
    z = hcat(x,y)
    
    xt = track(x)
    yt = track(y)
    zt = hcat(xt,yt)
    zt.grad = z
    accum!(xt)
    accum!(yt)
    @test xt.data ≈ z[:,1]
    @test yt.data ≈ z[:,2]


    # Simple Neural Network
    W1, b1 = track(rand(3,4)), track(rand(3))
    W2, b2 = track(rand(2,3)), track(rand(2))
    xs = [rand(4) |> track for _ in 1:10]
    ys = [rand(2) |> track for _ in 1:10]
    
    function forward(x, W1, b1, W2, b2)
        y1 = σ(W1*x + b1)
        y2 = W2*y1 + b2
    end
    
    function loss(xs, ys, model...)
        errs = hcat([forward(x,model...)-y for (x,y) in zip(xs,ys)]...)
        errs |> abs2 |> sum
    end

    dr = ReverseDiff.gradient(b1->loss(xs,ys,W1,b1,W2,b2), b1)[1]
    Base.abs2(x::Matrix) = abs2.(x)
    df = grad(central_fdm(5,1), b1->loss(data.(xs),data.(ys),W1.data,b1,W2.data,b2.data), b1.data)[1]
    @test dr ≈ df

    # make sure that reset works
    ReverseDiff.reset!.([W1,b1,W2,b2])
    xs = [rand(4) |> track for _ in 1:10]
    ys = [rand(2) |> track for _ in 1:10]
    df = grad(central_fdm(5,1), b1->loss(data.(xs),data.(ys),W1.data,b1,W2.data,b2.data), b1.data)[1]
    dr = ReverseDiff.gradient(b1->loss(xs,ys,W1,b1,W2,b2), b1)[1]
    @test dr ≈ df
end
