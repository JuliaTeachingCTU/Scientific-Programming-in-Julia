"""
	sample_all_installed_pkgs(path::AbstractString)

Returns root folders of all installed packages in the system. Package version is sampled.
"""
function sample_all_installed_pkgs(path::AbstractString)
	pkgs = readdir(path)
	# [rand(readdir(joinpath(path, p), join=true)) for p in pkgs] # sampling version
	[readdir(joinpath(path, p), join=true)[1] for p in pkgs if isdir(p)]    # deterministic version
end

"""
	filter_jl(path)

Recursively walks the directory structure to obtain all `.jl` files.
"""
filter_jl(path) = reduce(vcat, joinpath.(rootpath, filter(endswith(".jl"), files)) for (rootpath, dirs, files) in walkdir(path))

"""
	tokenize(jl_path)

Parses a ".jl" file located at `jl_path` and extracts all symbols and expression heads from the extracted AST.
"""
function tokenize(jl_path)
	_extract_symbols(x) = Symbol[]
	_extract_symbols(x::Symbol) = [x]
	function _extract_symbols(x::Expr) 
		if length(x.args) > 0
			Symbol.(vcat(x.head, reduce(vcat, _extract_symbols(arg) for arg in x.args)))
		else
			Symbol[]
		end
	end
	
	scode = "begin\n" * read(jl_path, String) * "end\n"
	try 
		code = Meta.parse(scode)
		_extract_symbols(code)
	catch e
		if ~isa(e, Meta.ParseError)
			rethrow(e)		
		end
		Symbol[]
	end
end
