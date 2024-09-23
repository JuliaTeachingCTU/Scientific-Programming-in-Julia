using Documenter

using Downloads: download
using Documenter.Writers: HTMLWriter
using DocumenterTools.Themes


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
pages = [
    "Home" => "index.md",
    "Installation" => "installation.md", 
    "Projects" => "projects.md",
    "1: Introduction" => [
        "Motivation" => "./lecture_01/motivation.md",
        "Basics" => "./lecture_01/basics.md",
        "Examples" => "./lecture_01/demo.md",
        "Outline" => "./lecture_01/outline.md",
        "Lab" => "./lecture_01/lab.md",
        "Homework" => "./lecture_01/hw.md",
    ],

    "2: The power of type system & multiple dispatch" => [
        "Lecture" => "./lecture_02/lecture.md",
        "Lab" => "./lecture_02/lab.md",
        "Homework" => "./lecture_02/hw.md",
    ],

    "3: Design patterns" => [
        "Lecture" => "./lecture_03/lecture.md",
        "Lab" => "./lecture_03/lab.md",
        "Homework" => "./lecture_03/hw.md",
    ],

    "4: Package development, unit tests & CI" => [
        "Lecture" => "./lecture_04/lecture.md",
        "Lab" => "./lecture_04/lab.md",
        "Homework" => "./lecture_04/hw.md",
    ],

    "5: Performance benchmarking" => [
        "Lecture" => "./lecture_05/lecture.md",
        "Lab" => "./lecture_05/lab.md",
        "Homework" => "./lecture_05/hw.md",
    ],

    "6: Lanuage introspection" => [
        "Lecture" => "./lecture_06/lecture.md",
        "Lab" => "./lecture_06/lab.md",
        "Homework" => "./lecture_06/hw.md",
    ],

    "7: Macros" => [
        "Lecture" => "./lecture_07/lecture.md",
        "Lab" => "./lecture_07/lab.md",
        "Homework" => "./lecture_07/hw.md",
    ],

    "8: Automatic differentiation" => [
        "Lecture" => "./lecture_08/lecture.md",
        "Lab" => "./lecture_08/lab.md",
        "Homework" => "./lecture_08/hw.md",
    ],

    "9: Intermediate representation" => [
        "Lecture" => "./lecture_09/lecture.md",
        "Lab" => "./lecture_09/lab.md",
    ],

    "10: Parallel programming" => [
        "Lecture" => "./lecture_10/lecture.md",
        "Lab" => "./lecture_10/lab.md",
        "Homework" => "./lecture_10/hw.md",
    ],

    "11: GPU programming" => [
        "Lecture" => "./lecture_11/lecture.md",
        "Lab" => "./lecture_11/lab.md",
    ],

    "12: Ordinary Differential Equations" => [
        "Lecture" => "./lecture_12/lecture.md",
        "Lab" => "./lecture_12/lab.md",
        "Homework" => "./lecture_12/hw.md",
    ],
]


makedocs(;
    authors = "JuliaTeachingCTU",
    # repo = "https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/{commit}{path}#{line}",
    sitename = "Scientific Programming in Julia",
    pagesonly = true,
    format = Documenter.HTML(;
        prettyurls = true,
        canonical = "https://JuliaTeachingCTU.github.io/Scientific-Programming-in-Julia",
        assets = ["assets/favicon.ico", "assets/onlinestats.css"],
        collapselevel = 1,
        ansicolor=true,
    ),
    pages
)

# deploydocs(;
#     repo = "github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia",
# )
