
function f(x,θ)
  α,β,γ,δ = θ
  x1,x2=x
   dx1 = α*x1 - β*x1*x2
   dx2 = δ*x1*x2 - γ*x2
  [dx1,dx2]
end

function solve(f,x0::AbstractVector,θ,dt,N)
  X = hcat([zero(x0) for i=1:N]...)
  X[:,1]=x0
  for t=1:N-1
     X[:,t+1]=X[:,t]+dt*f(X[:,t],θ)
  end
  X
end

θ0 = [0.1,0.2,0.3,0.2]
x0 = [1.0,1.0]
dt = 0.1
N  = 1000
X=solve(f,x0,θ0,dt,N)


using Plots

p=plot(X[1,:],xlabel="t",label="x",color=:blue)
p=plot!(X[2,:],xlabel="t",label="y",color=:red)


K=10
X0 = [x0 .+ 0.1*randn(2) for k=1:K]
Xens=[X=solve(f,X0[i],θ0,dt,N) for i=1:K]

for i=1:10
  p=plot!(Xens[i][1,:],label="",color=:blue)
  p=plot!(Xens[i][2,:],label="",color=:red)  
end

savefig("LV_ensemble.svg")

using StatsBase
Xm=mean(Xens)
Xstd = std(Xens)

struct GaussNum{T<:Real}
  μ::T
  σ::T
end
import Base: +, -, *, zero
+(x::GaussNum{T},a::T) where T =GaussNum(x.μ+a,x.σ)
+(a::T,x::GaussNum{T}) where T =GaussNum(x.μ+a,x.σ)
-(x::GaussNum{T},a::T) where T =GaussNum(x.μ-a,x.σ)
-(a::T,x::GaussNum{T}) where T =GaussNum(x.μ-a,x.σ)
*(x::GaussNum{T},a::T) where T =GaussNum(x.μ*a,a*x.σ)
*(a::T,x::GaussNum{T}) where T =GaussNum(x.μ*a,a*x.σ)


+(x1::GaussNum{T},x2::GaussNum{T}) where T =GaussNum(x1.μ+x2.μ, sqrt(x1.σ.^2 + x2.σ.^2))
-(x1::GaussNum{T},x2::GaussNum{T}) where T =GaussNum(x1.μ-x2.μ, sqrt(x1.σ.^2 + x2.σ.^2))
*(x1::GaussNum{T},x2::GaussNum{T}) where T =GaussNum(x1.μ*x2.μ, sqrt((x2.μ*x1.σ).^2 + (x1.μ * x2.σ).^2))
zero(::Type{GaussNum{T}}) where T =GaussNum(zero(T),zero(T))
zero(x::AbstractVector{T}) where T =[zero(T) for i=1:length(x)]

function MV(x::AbstractArray{GaussNum{T}}) where T
  M=similar(x,T)
  V=similar(x,T)
  for i=1:length(x)
    M[i]=x[i].μ
    V[i]=x[i].σ
  end
  (M,V)
end

GX=solve(f,[GaussNum(1.0,0.1),GaussNum(1.0,0.1)],[0.1,0.2,0.3,0.2],0.1,1000)
M,V=MV(GX)
plot(M',errorbar=V')
using Measurements
GX=solve(f,[1.0±0.1,1.0±0.1],[0.1,0.2,0.3,0.2],0.1,1000)
