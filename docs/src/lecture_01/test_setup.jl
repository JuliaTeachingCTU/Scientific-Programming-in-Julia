using InteractiveUtils: versioninfo
versioninfo()

println("-------------------------------------------------------------------------")
println("Julia started from terminal without args: ", length(ARGS) == 0 ? "✔" : "✗")
println("Running from the same folder as this script: ", isfile("./test_setup.jl") ? "✔" : "✗")
println("Running Julia 1.6.0 or above: ", VERSION >= v"1.6.0" ? "✔" : "✗")


envdir = "./L1Env/"
s = if !isdir(envdir)
	using Pkg

	Pkg.generate(envdir)
	Pkg.activate(envdir)
	Pkg.add("BenchmarkTools")
	true
else
	false
end

println("Installed test environment for later use: ", s ? "✔" : "?")

