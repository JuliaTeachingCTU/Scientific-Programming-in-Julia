using Scientific_Programming_in_Julia
using Documenter

DocMeta.setdocmeta!(Scientific_Programming_in_Julia, :DocTestSetup, :(using Scientific_Programming_in_Julia); recursive=true)

makedocs(;
    modules=[Scientific_Programming_in_Julia],
    authors="JuliaTeachingCTU",
    repo="https://github.com/JuliaTeachingCTU/Scientific_Programming_in_Julia.jl/blob/{commit}{path}#{line}",
    sitename="Scientific_Programming_in_Julia.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaTeachingCTU.github.io/Scientific_Programming_in_Julia.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaTeachingCTU/Scientific_Programming_in_Julia.jl",
)
