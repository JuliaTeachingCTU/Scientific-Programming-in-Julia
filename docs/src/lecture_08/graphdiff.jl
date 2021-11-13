using GraphRecipes
using LightGraphs
using Plots

default(size=(1000, 1000))

f = :(x * y + sin(x))

names = ["x", "y", "y₁ = sin", "y₂ = *", "y₃ = +", "z"]
g = LightGraphs.DiGraph(length(names))
add_edge!(g, 1, 3)  # x -> sin
add_edge!(g, 1, 4)	# x ->  *
add_edge!(g, 2, 4)	# y ->  *
add_edge!(g, 3, 5)	# sin(x) -> sin(x) + x*y
add_edge!(g, 4, 5)  # x*y -> sin(x) + x*y
add_edge!(g, 5, 6)  # sin(x) + x*y -> z
graphplot(adjacency_matrix(g), names = names)


vertices = Dict([
1 => (name = "x", 			x = 1, y = 0, color = :lightblue, ),
2 => (name = "y", 			x = 0, y = 1, color = :lightblue, ),
3 => (name = "y₁ = sin", 	x = 1, y = 2, color = :lightblue, ),
4 => (name = "y₂ = *", 		x = 0, y = 3, color = :lightblue, ),
5 => (name = "y₃ = +", 		x = 1, y = 4, color = :lightblue, ),
6 => (name = "z", 			x = 1, y = 5, color = :lightblue, ),
7 => (name = "∂z/∂y₃", 		x = 2.5, y = 4, color = :orange, ),
8 => (name = "∂y₃/∂y₂", 	x = 2.5, y = 3, color = :orange, ),
9 => (name = "∂z/∂y₂", 		x = 3.5, y = 3, color = :orange, ),
10 => (name = "∂y₃/∂y₁", 	x = 2.5, y = 2, color = :orange, ),
11 => (name = "∂z/∂y₁", 	x = 3.5, y = 2, color = :orange, ),
12 => (name = "∂y₂/∂y", 	x = 4.5, y = 1, color = :orange, ),
13 => (name = "∂y₂/∂x + ∂y₁/∂x", x = 4.5, y = 0, color = :orange, ),
14 => (name = "∂z/∂x", x = 6, y = 0, color = :orange, ),
])

n = length(vertices)
g = LightGraphs.DiGraph(n)
add_edge!(g, 1, 3)  # x -> sin
add_edge!(g, 1, 4)	# x ->  *
add_edge!(g, 2, 4)	# y ->  *
add_edge!(g, 3, 5)	# sin(x) -> sin(x) + x*y
add_edge!(g, 4, 5)  # x*y -> sin(x) + x*y
add_edge!(g, 5, 6)  # sin(x) + x*y -> z
add_edge!(g, 7, 5)	# ∂z/∂y₃ -> z
add_edge!(g, 8, 4)	# ∂y₃/∂y₂ -> y₂ = *
add_edge!(g, 9, 8)	# ∂z/∂y₂ -> ∂y₃/∂y₂
add_edge!(g, 9, 7)	# ∂z/∂y₂ -> ∂y₃/∂y₂
add_edge!(g, 10, 3)	# ∂y₃/∂y₁ -> y₁
add_edge!(g, 11, 10)# ∂z/∂y₁  -> ∂y₃/∂y₁
add_edge!(g, 11, 7) # ∂z/∂y₁  -> ∂z/∂y₃
add_edge!(g, 12, 2) # ∂y₂/∂y  -> y
add_edge!(g, 12, 9) # ∂y₂/∂y  -> y
add_edge!(g, 13, 1) # ∂y₂/∂x + ∂y₁/∂x  -> y
add_edge!(g, 14, 9) # "∂z/∂x"  -> ∂z/∂y₂
add_edge!(g, 14, 11) # "∂z/∂x"  -> ∂z/∂y₁
add_edge!(g, 14, 13) # "∂z/∂x"  -> ∂y₂/∂x + ∂y₁/∂x
graphplot(adjacency_matrix(g),
 names = [vertices[i].name for i in 1:n],
 x = 0.25 .* [vertices[i].x for i in 1:n],
 y = 0.25 .* [vertices[i].y for i in 1:n],
 curves=false,
 markercolor = [vertices[i].color for i in 1:length(vertices)],
)

n = 6
graphplot(adjacency_matrix(g)[1:n, 1:n],
 names = [vertices[i].name for i in 1:n],
 x = 0.25 .* [vertices[i].x for i in 1:n],
 y = 0.25 .* [vertices[i].y for i in 1:n],
 curves=false,
 markercolor = [vertices[i].color for i in 1:length(vertices)],
)

