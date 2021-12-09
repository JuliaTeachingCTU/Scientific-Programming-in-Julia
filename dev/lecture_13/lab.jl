using LinearAlgebra
using Statistics

θ = [0.1,GaussNum(0.2,0.1),0.3,0.2]
u0 = [GaussNum(1.0,0.1),GaussNum(1.0,0.1)]
tspan = (0.,100.)
dt = 0.1
prob = ODEProblem(f,tspan,u0,θ)


isuncertain(x::GaussNum) = x.σ!=0
isuncertain(x::Number) = false
sig(x::GaussNum) = x.σ
mu(x::GaussNum) = x.μ

#struct UncertainODEProblem{OP<:ODEProblem,S0<:AbstractMatrix,S<:AbstractMatrix,X<:AbstractMatrix,I,J}
#    prob::OP
#    √Σ::S0
#    Xp::X
#    u_idx::I
#    θ_idx::J
#    function UncertainODEProblem(prob::OP) where OP<:ODEProblem
#        u_idx = findall(isuncertain, prob.u0)
#        θ_idx = findall(isuncertain, prob.θ)
#
#        uσ = [u.σ for u in prob.u0[u_idx]]
#        θσ = [θ.σ for θ in prob.θ[θ_idx]]
#        √Σ = Diagonal(vcat(uσ,θσ))
#
#        n = (length(uσ)+length(θσ))*2
#        Qp = sqrt(n) * [I(n) -I(n)]
#        Xp = vcat(prob.u0,prob.θ) .+ √Σ*Qp
#        Σ  = √Σ * √Σ'
#
#        new{OP,typeof(√Σ),typeof(u_idx),typeof(θ_idx)}(prob,√Σ,u_idx,θ_idx)
#    end
#end
struct UncertainODEProblem{OP,US,I,J} <: AbstractODEProblem
    prob::OP
    U0::US
    u_idx::I
    θ_idx::J
    function UncertainODEProblem(prob::ODEProblem)
        u_idx = findall(isuncertain, prob.u0)
        θ_idx = findall(isuncertain, prob.θ)
        idx = vcat(u_idx, length(prob.u0) .+ θ_idx)

        n = length(idx)
        Qp = hcat(map(idx) do i
            q = zeros(length(prob.u0)+length(prob.θ))
            q[i] = sqrt(n)
            q
        end, map(idx) do i
            q = zeros(length(prob.u0)+length(prob.θ))
            q[i] = -sqrt(n)
            q
        end)
        Qp = reduce(hcat,Qp)
        Σ_ = Diagonal(vcat(sig.(prob.u0), sig.(prob.θ)))
        μ0 = vcat(mu.(prob.u0),mu.(prob.θ)) .+ Σ_*Qp
        #U0 = μ0 .± diag(Σ_)

        prob = ODEProblem(prob.f, prob.tspan, mu.(prob.u0), mu.(prob.θ))

        new{typeof(prob),typeof(μ0),typeof(u_idx),typeof(θ_idx)}(prob,μ0,u_idx,θ_idx)
    end
end

nr_uncertainties(p::UncertainODEProblem) = length(p.u_idx)+length(p.θ_idx)

struct UncertainODESolver{S<:ODESolver} <: ODESolver
    solver::S
end

# function get_xp(p::UncertainODEProblem,i::Int)
#     xp = p.Xp[:,i]
#     u  = xp[p.u_idx]
#     θ  = p.prob.θ
#     θ[p.θ_idx] .= xp[length(p.u_idx)+1:end]
#     (u,θ)
# end


function setmean!(p::UncertainODEProblem, x)
    nu = length(p.prob.u0)
    p.prob.θ .= x[nu .+ (1:length(p.prob.θ))]
    u = x[1:nu]
end

function (s::UncertainODESolver)(p::UncertainODEProblem, μs, t)
    N = nr_uncertainties(p)
    μs = map(1:N) do i
        u = setmean!(p, μs[:,i])
        u = s.solver(p.prob, u, t)[1]
        vcat(u, p.prob.θ)
    end
    μs = reduce(hcat, μs) 
    μ = mean(μs,dims=2)
    Σ = Matrix((μs .- μ)*(μs .- μ)'/N)
    σ = sqrt.(diag(Σ))
    σ[p.u_idx] .= 0
    σ[p.θ_idx .+ length(p.prob.u0)] .= 0
    #μ .± σ, t+1
    μs, t+1
end

function solve(p::UncertainODEProblem, solver::UncertainODESolver)
    t = p.prob.tspan[1]; u = p.U0
    us = [u]; ts = [t]
    while t < prob.tspan[2]
        (u,t) = solver(p, u, t)
        push!(us,u)
        push!(ts,t)
    end
    ts, reduce(hcat,us)
end



uprob = UncertainODEProblem(prob)
solver = UncertainODESolver(RK2(0.2))
t, X = solve(uprob,solver)

# function solve(f,x0::AbstractVector,sqΣ0, θ,dt,N)
#     n = length(x0)
#     n2 = 2*length(x0)
#     Qp = sqrt(n)*[I(n) -I(n)]
# 
#     X = hcat([zero(x0) for i=1:N]...)
#     S = hcat([zero(x0) for i=1:N]...)
#     X[:,1]=x0
#     Xp = x0 .+ sqΣ0*Qp
#     sqΣ = sqΣ0
#     Σ = sqΣ* sqΣ'
#     S[:,1]= diag(Σ)
#     for t=1:N-1
#         for i=1:n2 # all quadrature points
#           Xp[:,i].=Xp[:,i] + dt*f(Xp[:,i],θ)
#         end
#         mXp=mean(Xp,dims=2)
#         X[:,t+1]=mXp
#         Σ=Matrix((Xp.-mXp)*(Xp.-mXp)'/n2)
#         S[:,t+1]=sqrt.(diag(Σ))
#         # @show Σ
# 
#     end
#     X,S
# end








