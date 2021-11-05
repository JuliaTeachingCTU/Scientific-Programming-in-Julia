using Test
using Scientific_Programming_in_Julia
using FiniteDifferences

import Scientific_Programming_in_Julia.ReverseDiff
import Scientific_Programming_in_Julia.ReverseDiff.σ

@testset "ReverseDiff" begin
    A = rand(4,3)
    B = rand(3,2)
    f(X,Y) = sum(σ(X*Y))

    rgs = ReverseDiff.gradient(f,A,B)
    fgs = grad(central_fdm(5,1), f, A, B)
    for (rg,fg) in zip(rgs,fgs)
        @test rg ≈ fg
    end
end

@testset "ReverseDiff Neural Network" begin
    W1, b1 = track(rand(3,4)), track(rand(3))
    W2, b2 = track(rand(2,3)), track(rand(2))
    
    function forward(x, W1, b1, W2, b2)
        y1 = σ(W1*x + b1)
        y2 = W2*y1 + b2
    end
    
    function loss(x, y, model...)
        ȳ = forward(x, model...)
        sum(abs2(y-ȳ))
    end

    x = rand(4) |> track
    y = rand(2) |> track
    l = loss(x, y, W1, b1, W2, b2)
    l.grad = 1.0
    dr = accum!(b1)
    df = grad(central_fdm(5,1), b1->loss(x,y,W1,b1,W2,b2), b1)[1].data
    @test dr ≈ df
end
