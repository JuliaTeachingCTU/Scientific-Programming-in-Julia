using InteractiveUtils: versioninfo
versioninfo()

println("-------------------------------------------------------------------------")
println("Julia started from terminal without args: ", length(ARGS) == 0 ? "✔" : "✗")
println("Running from the same folder as this script: ", isfile("./test_setup.jl") ? "✔" : "✗")
println("Running Julia 1.6.0 or above: ", VERSION >= v"1.6.0" ? "✔" : "✗")

name = try
	readchomp(`git config user.name`)
catch
	""
end

mail = try
	readchomp(`git config user.email`)
catch
	""
end


println("Git Config Username: ", length(name) > 0 ? "✔" : "✗")
println("Git Config Email: ", length(mail) > 0 ? "✔" : "✗")

s = try
	using Pkg
	Pkg.activate(".")
	Pkg.add("BenchmarkTools")
	true
catch e
	@warn e
	false
end

println("Installed test environment for later use: ", s ? "✔" : "?")

