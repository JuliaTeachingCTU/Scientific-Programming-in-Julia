using Scientific_Programming_in_Julia
using Documenter

using Downloads: download
using Documenter.Writers: HTMLWriter
using DocumenterTools.Themes

DocMeta.setdocmeta!(
    Scientific_Programming_in_Julia,
    :DocTestSetup,
    :(using Scientific_Programming_in_Julia);
    recursive = true
)

# download and compile theme
assetsdir(args...) = joinpath(@__DIR__, "src", "assets", args...)
site = "https://github.com/JuliaTeachingCTU/JuliaCTUGraphics/raw/main/"
force = true

mkpath(assetsdir("themes"))
mv(download("$(site)logo/CTU-logo-dark.svg"), assetsdir("logo-dark.svg"); force)
mv(download("$(site)logo/CTU-logo.svg"), assetsdir("logo.svg"); force)
mv(download("$(site)icons/favicon.ico"), assetsdir("favicon.ico"); force)

for theme in ["light", "dark"]
    mktemp(@__DIR__) do path, io
        write(io, join([
            read(joinpath(HTMLWriter.ASSETS_THEMES, "documenter-$(theme).css"), String),
            read(download("$(site)assets/lectures-$(theme).css"), String)
        ], "\n"))
        Themes.compile(
            path,
            joinpath(@__DIR__, assetsdir("themes", "documenter-$(theme).css"))
        )
    end
end

# documentation
makedocs(;
    modules = [Scientific_Programming_in_Julia],
    authors = "JuliaTeachingCTU",
    repo = "https://github.com/JuliaTeachingCTU/Scientific_Programming_in_Julia.jl/blob/{commit}{path}#{line}",
    sitename = "Scientific Programming in Julia",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JuliaTeachingCTU.github.io/Scientific_Programming_in_Julia.jl",
        assets = ["assets/favicon.ico"],
    ),
    pages = [
        "Home" => "index.md",
        "How To ..." => "howto.md",
    ],
)

deploydocs(;
    repo = "github.com/JuliaTeachingCTU/Scientific_Programming_in_Julia.jl",
)
