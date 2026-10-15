# Compatibility loader for notebooks run directly from this folder.
# For normal package use, activate this project and write `using Ecosystem`.
include(joinpath(@__DIR__, "src", "Ecosystem.jl"))
