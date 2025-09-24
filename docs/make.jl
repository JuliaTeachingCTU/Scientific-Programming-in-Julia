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

deploydocs(;
    repo=repo,
    target="build",
    devbranch="main",
    branch="gh-pages",
    push_preview=true,
)
