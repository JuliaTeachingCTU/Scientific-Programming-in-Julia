# Lab 13 - Data-driven Differential Equations

In this lab you will implement your own _*Physics Informed Neural Network*_ (PINN)
(discussed in [lecture 13](@ref lec13) and an improved uncertainty propagation
based on the cubature rules from [lecture 12](@ref lec13).

_*Before your start*_: Please, install the necessary packages for this lab and
let them precompile while you familiarize yourself with the PINN implementation.
```julia
(@1.6) pkg> add Flux ForwardDiff Optim GalacticOptim Plots
```
