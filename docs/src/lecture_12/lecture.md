Ordinary Differencial equation

Differential equations are commonly used in science to describe many aspects of the physical world, ranging from dynamical systems, curves in space, to a complex multi-physics phenomena. 

As an example, consider a simple non-linear ordinary differential equation:

```math
\begin{align}
\dot{x}&=\alpha x-\beta xy,\\\dot{y}&=-\delta y+\gamma xy, 
\end{align}
```

Which describes behavior of a predator-pray models in continuous times:
 - x is the population of prey (sheep),
 - y is the population of predator (wolfes)
 - derivatives represent instantaneous growth rates of the populations
 - ``t`` is the time and ``\alpha, \beta, \gamma, \delta`` are parameters.

Can be written in vector arguments ``\mathbf{x}=[x,y]``:
```math
\frac{d\mathbf{x}}{dt}=f(\mathbf{x},\theta)
```
with arbitrary function ``f`` with vector of parmeters ``\theta``.


The first steps we may want to do with an ODE is to see it evolution in time. The most simple approach is to discretize the time axis into steps:
``t = [t_1, t_2, t_3, \ldots t_T]``
and evaluate solution at these points.

Replacing derivatives by differences:
```math
\dot x \leftarrow \frac{x_t-x_{t-1}}{\Delta t}
```
we can derive a general  scheme (Euler solution):
```math
\mathbf{x}_t = \mathbf{x}_{t-1} + \Delta{}t f(\mathbf{x}_t,\theta)
```
which can be written genericaly in julia :
```julia

function f(x,θ)
  α,β,γ,δ = θ
  x1,x2=x
   dx1 = α*x1 - β*x1*x2
   dx2 = δ*x1*x2 - γ*x2
  [dx1,dx2]
end

function solve(f,x0::AbstractVector,θ,dt,N)
  X = zeros(length(x0),N)
  X[:,1]=x0
  for t=1:N-1
     X[:,t+1]=X[:,t]+dt*f(X[:,t],θ)
  end
  X
end
```

Is trivial but works:
![](lotka.svg)

Note that the method was dispatched on abstract vector. The reason is that ODE does not only 

- check Chris Raucausacc latest blog
  https://julialang.org/blog/2021/10/DEQ/
