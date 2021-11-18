using TikzGraphs
using TikzPictures
using GraphRecipes
using LightGraphs
using Plots

default(size=(1000, 1000))

f = :(x * y + sin(x))

vertices = Dict([
1 => (name = "x", 			x = 1, y = 0, color = :lightblue, ),
2 => (name = "y", 			x = 0, y = 1, color = :lightblue, ),
3 => (name = "h₁ = sin", 	x = 1, y = 2, color = :lightblue, ),
4 => (name = "h₂ = *", 		x = 0, y = 3, color = :lightblue, ),
5 => (name = "h₃ = +", 		x = 1, y = 4, color = :lightblue, ),
6 => (name = "z", 			x = 1, y = 5, color = :lightblue, ),
7 => (name = "∂z/∂h₃", 		x = 2.5, y = 4, color = :orange, ),
8 => (name = "∂h₃/∂h₂", 	x = 2.5, y = 3, color = :orange, ),
9 => (name = "∂z/∂h₂", 		x = 3.5, y = 3, color = :orange, ),
10 => (name = "∂h₃/∂h₁", 	x = 2.5, y = 2, color = :orange, ),
11 => (name = "∂z/∂h₁", 	x = 3.5, y = 2, color = :orange, ),
12 => (name = "∂h₂/∂y", 	x = 4.5, y = 1, color = :orange, ),
13 => (name = "∂h₂/∂x + ∂h₁/∂x", x = 4.5, y = 0, color = :orange, ),
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
#add_edge!(g, 5, 7,)	# ∂z/∂h₃ -> z
#add_edge!(g, 4, 8,)	# ∂h₃/∂h₂ -> h₂ = *
#add_edge!(g, 8, 9,)	# ∂z/∂h₂ -> ∂h₃/∂h₂
#add_edge!(g, 7, 9,)	# ∂z/∂h₂ -> ∂h₃/∂h₂
#add_edge!(g, 3, 10,)	# ∂h₃/∂h₁ -> h₁
#add_edge!(g, 10, 11,)# ∂z/∂h₁  -> ∂h₃/∂h₁
#add_edge!(g, 7, 11,) # ∂z/∂h₁  -> ∂z/∂h₃
#add_edge!(g, 2, 12,) # ∂h₂/∂y  -> y
#add_edge!(g, 9, 12,) # ∂h₂/∂y  -> y
#add_edge!(g, 1, 13,) # ∂h₂/∂x + ∂h₁/∂x  -> y
#add_edge!(g, 9, 14,) # "∂z/∂x"  -> ∂z/∂h₂
#add_edge!(g, 11, 14,) # "∂z/∂x"  -> ∂z/∂h₁
#add_edge!(g, 13, 14,) # "∂z/∂x"  -> ∂h₂/∂x + ∂h₁/∂x
# graphplot(adjacency_matrix(g),
# 	names = [vertices[i].name for i in 1:n],
# 	x = 0.25 .* [vertices[i].x for i in 1:n],
# 	y = 0.25 .* [vertices[i].y falseor i in 1:n],
# 	curves=false,
# 	markercolor = [vertices[i].color for i in 1:n],
# )


for n in [6, 7, 9, 11, 13, 14]
    #graphplot(adjacency_matrix(g)[1:n, 1:n],
	#	names = [vertices[i].name for i in 1:n],
	#	x = 0.25 .* [vertices[i].x for i in 1:n],
	#	y = 0.25 .* [vertices[i].y for i in 1:n],
	#	curves=false,
	#	markercolor = [vertices[i].color for i in 1:n],
	#)
	names = [vertices[i].name for i in 1:length(vertices)]
    t = TikzGraphs.plot(g, names)
    TikzPictures.save(SVG("graphdiff_$(n).svg"),t)
end
