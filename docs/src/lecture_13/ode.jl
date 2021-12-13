
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
θ1 =[[θ0[1]+0.01randn();θ0[2:end]] for k=1:K]
Xens=[X=solve(f,X0[i],θ1[i],dt,N) for i=1:K]

for i=1:K
  p=plot!(Xens[i][1,:],label="",color=:blue)
  p=plot!(Xens[i][2,:],label="",color=:red)  
end

MXe=mean(Xens)
SXe=std(Xens)
savefig("LV_ensemble.svg")
plot(MXe[1,1:30:end],label="x",color=:blue,errorbar=SXe[1,1:30:end])
plot!(MXe[2,1:30:end],label="y",color=:red,errorbar=SXe[2,1:30:end])



using LinearAlgebra
function solve(f,x0::AbstractVector,Σ0, θ,dt,N)
   n = length(x0)
   n2 = 2*length(x0)
   Qp = sqrt(n)*[I(n) -I(n)]

  X = hcat([zero(x0) for i=1:N]...)
  S = hcat([zero(x0) for i=1:N]...)
  X[:,1]=x0
  Σ=Hermitian(Σ0)
  sqΣ = cholesky(Σ).L
  Xp = x0 .+ sqΣ*Qp
  S[:,1]= diag(Σ)
  for t=1:N-1
    # if rem(t,Nr)==0
    #   Xp .= X[:,t] .+ sqΣ * Qp
    # end
    for i=1:n2 # all quadrature points
      Xp[:,i].=Xp[:,i] + [dt*f(Xp[1:2,i],[Xp[3,i];θ[2:end]]);0]
    end
    mXp=mean(Xp,dims=2)
    X[:,t+1]=mXp
    Σ=Matrix((Xp.-mXp)*(Xp.-mXp)'/n2)
    S[:,t+1]=sqrt.(diag(Σ))
    # @show Σ
  end
  X,S,Xp
end

## Extension to arbitrary 

QX,QS=solve(f,[1.0,1.0,0.1],diagm([0.1,0.1,0.01]),θ0,0.1,1000)
plot(QX[1,1:30:end],label="x",color=:blue,errorbar=QS[1,1:30:end])
plot!(QX[2,1:30:end],label="y",color=:red,errorbar=QS[2,1:30:end])

savefig("LV_Quadrics.svg")

function filter(f,x0::AbstractVector,Σ0, θ,dt,Ne,Y,σy)
  XX=[]
  SS=[]
  Σ=Σ0
  x=x0
  for t=1:length(Y)
    @show x
    @show eigen(Σ)
    Xt,St,Xp=solve(f,x,Σ,θ,dt,Ne) # prediction
    Yp = Xp[1,:] # measure only the first variable
    mYp = mean(Yp)
    mXp = mean(Xp,dims=2)
    SYp = cov(Yp)+σy
    C = mean([(Xp[:,i].-mXp)*(Yp[i].-mYp) for i=1:6])
    G = C*inv(SYp)
    x = vec(Xt[:,end] + G*(Y[t]-mYp))
    Σ = cov(Xp,dims=2) - G*SYp*G'
    push!(XX,Xt)
    push!(SS,St)
  end
  XX,SS
end



Y = X[1,100:100:end]
Xh,Sh=filter(f,[1.0,1.0,0.1],diagm([0.1,0.1,0.01].^2),θ0,0.1,100,Y,0.01)
XF=hcat(Xh...)
SF=hcat(Sh...)

step=10
plot([1:step:size(XF,2)],XF[1,1:step:end],label="x",color=:blue,errorbar=SF[1,1:step:end])
plot!([1:step:size(XF,2)],XF[2,1:step:end],label="y",color=:red,errorbar=SF[2,1:step:end])
scatter!([100:100:size(XF,2)],Y,label="measured",marker=:xcross)