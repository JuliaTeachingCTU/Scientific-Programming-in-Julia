using Documenter, DocumenterVitepress
using Documenter.Remotes

using Scientific_Programming_in_Julia

# utilities
function add_prefix(prefix::S, pair::Pair{S,T}) where {S<:AbstractString,T}
    key, val = pair
    if isa(val, AbstractString)
        return key => joinpath(prefix, val)
    else
        return key => add_prefix(prefix, val)
    end
end

function add_prefix(prefix::AbstractString, pairs::AbstractVector{<:Pair})
    return add_prefix.(prefix, pairs)
end

# pages
pages = [
    "Home" => "index.md",
    "Tutorials" => add_prefix("./tutorials", [
        "Installation" => "installation.md",
    ]),
    "Projects" => add_prefix("./projects", [
        "Requirements" => "requirements.md",
        "Potential projects" => "projects.md",
    ]),
    "Lectures" => add_prefix("./lectures", [
        "Outline" => "outline.md",
        "1: Introduction" => add_prefix("lecture_01", [
            "Motivation" => "motivation.md",
            "Basics" => "basics.md",
            "Examples" => "demo.md",
            "Lab" => "lab.md",
            "Homework" => "hw.md",
        ]),
        "2: The power of type system & multiple dispatch" => add_prefix("lecture_02", [
            "Lecture" => "lecture.md",
            "Lab" => "lab.md",
            "Homework" => "hw.md",
        ]),
        "3: Design patterns" => add_prefix("lecture_03", [
            "Lecture" => "lecture.md",
            "Lab" => "lab.md",
            "Homework" => "hw.md",
        ]),
        "4: Package development, unit tests & CI" => add_prefix("lecture_04", [
            "Lecture" => "lecture.md",
            "Lab" => "lab.md",
            "Homework" => "hw.md",
        ]),
        # "5: Performance benchmarking" => add_prefix("lecture_05", [
        #     "Lecture" => "lecture.md",
        #     "Lab" => "lab.md",
        #     "Homework" => "hw.md",
        # ]),
        # "6: Lanuage introspection" => add_prefix("lecture_06", [
        #     "Lecture" => "lecture.md",
        #     "Lab" => "lab.md",
        #     "Homework" => "hw.md",
        # ]),
        # "7: Macros" => add_prefix("lecture_07", [
        #     "Lecture" => "lecture.md",
        #     "Lab" => "lab.md",
        #     "Homework" => "hw.md",
        # ]),
        # "8: Automatic differentiation" => add_prefix("lecture_08", [
        #     "Lecture" => "lecture.md",
        #     "Lab" => "lab.md",
        #     "Homework" => "hw.md",
        # ]),
        # "9: Intermediate representation" => add_prefix("lecture_09", [
        #     "Lecture" => "lecture.md",
        #     "Lab" => "lab.md",
        # ]),
        # "10: Parallel programming" => add_prefix("lecture_10", [
        #     "Lecture" => "lecture.md",
        #     "Lab" => "lab.md",
        #     "Homework" => "hw.md",
        # ]),
        # "11: GPU programming" => add_prefix("lecture_11", [
        #     "Lecture" => "lecture.md",
        #     "Lab" => "lab.md",
        # ]),
        # "12: Ordinary Differential Equations" => add_prefix("lecture_12", [
        #     "Lecture" => "lecture.md",
        #     "Lab" => "lab.md",
        #     "Homework" => "hw.md",
        # ]),
    ]),
]

# documentation
organisation = "JuliaTeachingCTU"
repository = "Scientific-Programming-in-Julia"
repo = Remotes.GitHub(organisation, repository)

makedocs(;
    modules=[Scientific_Programming_in_Julia],
    authors=organisation,
    repo=repo,
    sitename="Scientific Programming in Julia",
    format=DocumenterVitepress.MarkdownVitepress(
        repo=Remotes.repourl(repo),
        md_output_path=".", # local build only
        build_vitepress=false, # local build only
    ),
    pages=pages,
    warnonly=true,
    clean=false # local build only
)

deploydocs(;
    repo=repo,
    target="build",
    devbranch="main",
    branch="gh-pages",
    push_preview=true,
)
