
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


K=100
X0 = [x0 .+ 0.1*randn(2) for k=1:K]
Xens=[X=solve(f,X0[i],θ0,dt,N) for i=1:K]

for i=1:K
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

# TODO
# sin(x::GaussNum)= @uncertain sin(x)


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
plot(M')
plot(M[1,1:30:end],errorbar=V[1,1:30:end],label="x",color=:blue)
plot!(M[2,1:30:end],errorbar=V[2,1:30:end],label="y",color=:red)

savefig("LV_GaussNum.svg")

GX=solve(f,[GaussNum(1.0,0.1),GaussNum(1.0,0.1)],[GaussNum(0.1,0.1),0.2,0.3,0.2],0.1,1000)
M,V=MV(GX)
plot(M')
plot(M[1,1:30:end],errorbar=V[1,1:30:end],label="x",color=:blue)
plot!(M[2,1:30:end],errorbar=V[2,1:30:end],label="y",color=:red)

savefig("LV_GaussNum2.svg")

using Measurements
MX=solve(f,[1.0±0.1,1.0±0.1],[0.1,0.2,0.3,0.2],0.1,1000)
plot(MX[1,1:30:end],label="x",color=:blue)
plot!(MX[2,1:30:end],label="y",color=:red)

savefig("LV_Measurements.svg")

MX=solve(f,[1.0±0.1,1.0±0.1],[0.1±0.01,0.2±0.01,0.3±0.01,0.2±0.01],0.1,1000)
plot(MX[1,1:30:end],label="x",color=:blue)
plot!(MX[2,1:30:end],label="y",color=:red)

savefig("LV_Measurements2.svg")



# Plot receipe
# plot(Vector{GaussNum})

using LinearAlgebra
function solve(f,x0::AbstractVector,sqΣ0, θ,dt,N,Nr)
   n = length(x0)
   n2 = 2*length(x0)
   Qp = sqrt(n)*[I(n) -I(n)]

  X = hcat([zero(x0) for i=1:N]...)
  S = hcat([zero(x0) for i=1:N]...)
  X[:,1]=x0
  Xp = x0 .+ sqΣ0*Qp
  sqΣ = sqΣ0
  Σ = sqΣ* sqΣ'
  S[:,1]= diag(Σ)
  for t=1:N-1
    if rem(t,Nr)==0
      Xp .= X[:,t] .+ sqΣ * Qp
    end
    for i=1:n2 # all quadrature points
      Xp[:,i].=Xp[:,i] + dt*f(Xp[:,i],θ)
    end
    mXp=mean(Xp,dims=2)
    X[:,t+1]=mXp
    Σ=Matrix((Xp.-mXp)*(Xp.-mXp)'/n2)
    S[:,t+1]=sqrt.(diag(Σ))
    # @show Σ

    sqΣ = cholesky(Σ).L

  end
  X,S
end

## Extension to arbitrary 

QX,QS=solve(f,[1.0,1.0],(0.1)*I(2),θ0,0.1,1000,1)
plot(QX[1,1:30:end],label="x",color=:blue,errorbar=QS[1,1:30:end])
plot!(QX[2,1:30:end],label="y",color=:red,errorbar=QS[2,1:30:end])

savefig("LV_Quadrics.svg")