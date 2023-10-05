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
lecture_01 = [
    "Motivation" => "./lecture_01/motivation.md",
    "Basics" => "./lecture_01/basics.md",
    "Examples" => "./lecture_01/demo.md",
    "Outline" => "./lecture_01/outline.md",
    "Lab" => "./lecture_01/lab.md",
    "Homework" => "./lecture_01/hw.md",
]

lecture_02 = [
    "Lecture" => "./lecture_02/lecture.md"
    "Lab" => "./lecture_02/lab.md"
    "Homework" => "./lecture_02/hw.md"
]

lecture_03 = [
    "Lecture" => "./lecture_03/lecture.md"
]

lecture_04 = [
    "Lecture" => "./lecture_04/lecture.md"
]


makedocs(;
    modules = [Scientific_Programming_in_Julia],
    authors = "JuliaTeachingCTU",
    repo = "https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/{commit}{path}#{line}",
    sitename = "Scientific Programming in Julia",
    format = Documenter.HTML(;
        prettyurls = true,
        canonical = "https://JuliaTeachingCTU.github.io/Scientific-Programming-in-Julia",
        assets = ["assets/favicon.ico", "assets/onlinestats.css"],
        collapselevel = 1,
        ansicolor=true,
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md", 
        "Projects" => "projects.md",
        "1: Introduction" => lecture_01,
        "2: The power of Type System & multiple dispatch" => lecture_02,
        "3: Design Patterns" => lecture_03,
        "4: Package development, Unit test & CI" => lecture_04,
    ],
)

deploydocs(;
    repo = "github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia",
)
