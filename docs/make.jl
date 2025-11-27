using Documenter, DocumenterVitepress
using Documenter.Remotes

using Scientific_Programming_in_Julia

# This is needed for live preview 
if get(ENV, "VITREPRESS_LIVE_PREVIEW", "false") == "true"
    VITREPRESS_KWARGS = (;
        md_output_path=".",
        build_vitepress=false,
    )
    MAKEDOCS_KWARGS = (; clean=false,)
else
    VITREPRESS_KWARGS = (;)
    MAKEDOCS_KWARGS = (;)
end

@show VITREPRESS_KWARGS
@show MAKEDOCS_KWARGS

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
    "Installation" => "tutorials/installation.md",
    "Lectures" => add_prefix("./lectures", [
        "Outline" => "outline.md",
        "1: Introduction" => add_prefix("lecture_01", [
            "Motivation" => "motivation.md",
            "Basics" => "basics.md",
            "Examples" => "demo.md",
        ]),
        "2: The power of type system & multiple dispatch" => "lecture_02/lecture.md",
        "3: Design patterns" => "lecture_03/lecture.md",
        "4: Package development, unit tests & CI" => "lecture_04/lecture.md",
        "5: Performance benchmarking" => "lecture_05/lecture.md",
        "6: Language introspection" => "lecture_06/lecture.md",
        "7: Macros" => "lecture_07/lecture.md",
        "8: Automatic differentiation 1" => "lecture_08/lecture.md",
        "9: Automatic differentiation 2" => "lecture_09/lecture_v2.md"
        "X: Manipulating Intermediate Represenation (IR)" => "lecture_09/lecture_v1.md"
    ]),
    "Labs" => add_prefix("./lectures", [
        "1: Introduction" => "lecture_01/lab.md",
        "2: The power of type system & multiple dispatch" => "lecture_02/lab.md",
        "3: Design patterns" => "lecture_03/lab.md",
        "4: Package development, unit tests & CI" => "lecture_04/lab.md",
        "5: Performance benchmarking" => "lecture_05/lab.md",
        "6: Language introspection" => "lecture_06/lab.md",
        "7: Macros" => "lecture_07/lab.md",
        "8: Automatic differentiation 1" => "lecture_08/lab.md",
        "9: Custom Rules For Differentiation" => "lecture_09/lab.md"
    ]),
    "Homeworks" => add_prefix("./lectures", [
        "1: Introduction" => "lecture_01/hw.md",
        "2: The power of type system & multiple dispatch" => "lecture_02/hw.md",
        "3: Design patterns" => "lecture_03/hw.md",
        "4: Package development, unit tests & CI" => "lecture_04/hw.md",
        "5: Performance benchmarking" => "lecture_05/hw.md",
        "6: Language introspection" => "lecture_06/hw.md",
        "7: Macros" => "lecture_07/hw.md",
        "8: Automatic differentiation 1" => "lecture_08/hw.md"
    ]),

    "Projects" => add_prefix("./projects", [
        "Requirements" => "requirements.md",
        "Potential projects" => "projects.md",
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
    format=DocumenterVitepress.MarkdownVitepress(;
        repo=Remotes.repourl(repo),
        VITREPRESS_KWARGS...,
    ),
    pages=pages,
    warnonly=true,
    MAKEDOCS_KWARGS...,
)

DocumenterVitepress.deploydocs(;
    repo=replace(Remotes.repourl(repo), "https://" => ""),
    target=joinpath(@__DIR__, "build"),
    devbranch="2025W",
    branch="gh-pages",
    push_preview=true,
)