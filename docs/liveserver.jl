#!/usr/bin/env julia

# Root of the repository
const repo_root = dirname(@__DIR__)

# Make sure docs environment is active and instantiated
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Communicate with docs/make.jl that we are running in live mode
push!(ARGS, "liveserver")

# Run LiveServer.servedocs(...)
import LiveServer
LiveServer.servedocs(;
    # Documentation root where make.jl and src/ are located
    foldername = joinpath(repo_root, "docs"),
    skip_dirs = [
        # exclude assets folder because it is modified by docs/make.jl
        joinpath("docs", "src", "assets")
    ],
)
