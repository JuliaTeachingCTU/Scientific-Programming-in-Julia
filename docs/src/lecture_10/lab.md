# [Lab 10: Parallel computing](@id parallel_lab)

# Introduction
- We don't think in parallel
- We learn to write and reason about programs serially
- The desire for parallelism often comes _after_ you've written your algorithm (and found it too slow!)
- We are always on the lookout for the end of Moore's Law, but so far the transistor count stays on the exponential trend.
- The number of cores is increasing
- CPUs' complexity is increasing with the addition of more sophisticated branch predictors
![42-cpu-trend](./42-years-processor-trend.png)
Image source[^1]

[^1]: Performance metrics trend of CPUs in the last 42years: [https://www.karlrupp.net/2018/02/42-years-of-microprocessor-trend-data/](https://www.karlrupp.net/2018/02/42-years-of-microprocessor-trend-data/)


!!! warning "Shortcomings of parallelism"
	Parallel computing brings its own set of problems and not an insignificant overhead with data manipulation and communication, therefore try always to optimize your serial code as much as you can before advancing to parallel acceleration.
	- parallel programs are hard to debug

## Distributed/multi-processing
Wrapping our heads around having multiple running Julia processes
- how to start
- how to load code
- how to send code there and back again
- convenience pkgs

There are two ways how to start multiple julia processes
- by adding processes using cmd line argument `-p ##`
```bash
julia -p 4
```
- by adding processes after startup using the `addprocs(##)` function from std library `Distributed`
```julia
using Distributed
addprocs(4) # returns a list of ids of individual processes
```

The result shown in a process manager such as `htop`:
```
.../julia-1.6.2/bin/julia --project                                                                                     
.../julia-1.6.2/bin/julia -Cnative -J/home/honza/Apps/julia-1.6.2/lib/julia/sys.so -g1 --bind-to 127.0.0.1 --worker
.../julia-1.6.2/bin/julia -Cnative -J/home/honza/Apps/julia-1.6.2/lib/julia/sys.so -g1 --bind-to 127.0.0.1 --worker
.../julia-1.6.2/bin/julia -Cnative -J/home/honza/Apps/julia-1.6.2/lib/julia/sys.so -g1 --bind-to 127.0.0.1 --worker
.../julia-1.6.2/bin/julia -Cnative -J/home/honza/Apps/julia-1.6.2/lib/julia/sys.so -g1 --bind-to 127.0.0.1 --worker
```

Both of these result in a running of 5 processes in total - 1 controller, 4 workers - with their respective ids accessible via `myid()` function call. Note that the controller process has always id 1 and other processes are assigned subsequent integers, see for yourself with `@everywhere` macro, which runs easily code on all or a subset of processes.
```julia
@everywhere println(myid())
@everywhere [2,3] println(myid()) # select a subset of workers
```

As we have seen from the `htop/top` output, added processes start with specific cmd line arguments, however they are not shared with any aliases that we may have defined, e.g. `julia` ~ `julia --project=.`. Therefore in order to use an environment, we have to first activate it on all processes
```julia
@everywhere begin
  using Pkg; Pkg.activate(@__DIR__) # @__DIR__ equivalent to a call to pwd()
  Pkg.instantiate(); Pkg.precompile() # this should not be necessary to call everywhere
end
```
We can load files on all processes `-L` has to include

There are generally two ways of working with multiple processes
- using low level functionality - we specify what/where is loaded, what/where is being run and when we fetch results
	+ `@everywhere` macro
	+ `@spawnat` macro
	+ `fetch` function
	+ `myid` function

- using high level functionality - define only simple functions and apply them on distributed data structures 
	+ `DistributedArrays`' `DArray`s
	+ `pmap`

### Sum with processes
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Write an aggregated sum with multiprocessing/distributed.
- without `DistributedArrays`
- with `DistributedArrays`

**HINTS**:
- for base functionality `DistributedArrays`' functionality of distributing preexisting array (`distribute` function) works well
- you have to consider option that the array may not fit into memory

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
using BenchmarkTools
a = rand(10^7);
using Distributed
# using DistributedArrays
addprocs(4)
@everywhere using DistributedArrays
adist = distribute(a)
j_bench_base = @benchmark sum($a)
j_bench_dist = @benchmark sum($adist)
```

```julia
function mysum_dist(a::DArray)
    r = Array{Future}(undef, length(procs(a)))
    for (i, id) in enumerate(procs(a))
        r[i] = @spawnat id sum(localpart(a))
    end
    return sum(fetch.(r))
end
j_bench_hand_dist = @benchmark mysum_dist($adist)
```



```@raw html
</p></details>
```

### Distributed file processing
`Distributed` is often used in processing of files, such as the commonly encountered `mapreduce` jobs with technologies like [`Hadoop`](https://hadoop.apache.org/), [`Spark`](http://spark.apache.org/), where the files live on a distributed file system and a typical job requires us to map over all the files and gather some statistics such as histograms, sums and others. We will simulate this situation with the Julia's pkg codebase, which on a typical user installation can contain up to hundreds of thousand of `.jl` files (depending on how extensively one uses Julia).

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```

Write a distributed pipeline for computing a histogram of symbols found in AST by parsing Julia source files in your `.julia/packages/` directory. We have already implemented most of the code that you will need (available as source code [here](source code)). **TODO**

Your job is to write the `map` and `reduce` steps, that will gather the dictionaries from different workers. There are two ways to map
- either over directories inside `.julia/packages/`
- or over all files obtained by concatenation of `filter_jl` outputs (*NOTE* that this might not be possible if the listing itself is expensive - speed or memory requirements)
Measure if the speed up scales linearly with the number of processes by restricting the number of workers inside a pmap.

**HINTS**:
- either load `./pkg_processing.jl` on startup with `-L` and `-p` options or `include("./pkg_processing.jl")` inside `@everywhere`
- try writing sequential version first
- use `pmap` to easily iterate in parallel over a collection - the result should be an array of histogram, which has to be merged on the controller node

**BONUS**:
What is the most frequent symbol in your codebase?

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

Let's implement first a sequential version as it is much easier to debug.
```julia
include("./pkg_processing.jl")

using ProgressMeter
function sequential_histogram(path)
	h = Dict{Symbol, Int}()
	@showprogress for pkg_dir in sample_all_installed_pkgs(path)
		for jl_path in filter_jl(pkg_dir)
			syms = tokenize(jl_path)
			for s in syms
				v = get!(h, s, 0)
				h[s] += 1
			end
		end
	end
	h
end
path = joinpath(DEPOT_PATH[1], "packages") # usually the first entry
@elapsed h = sequential_histogram(path)
```

First we try to distribute over package folders. 
```julia
using Distributed
addprocs(8)

@everywhere begin
  using Pkg; Pkg.activate(@__DIR__)
  # we have to realize that the code that workers have access to functions we have defined
  include("./pkg_processing.jl") 
end

"""
	merge_with!(h1, h2)

Merges count dictionary `h2` into `h1` by adding the counts.
"""
function merge_with!(h1, h2)
	for s in keys(h2)
		get!(h1, s, 0)
		h1[s] += h2[s]
	end
	h1
end

using ProgressMeter
function distributed_histogram(path)
	r = @showprogress pmap(sample_all_installed_pkgs(path)) do pkg_dir
		h = Dict{Symbol, Int}()
		for jl_path in filter_jl(pkg_dir)
			syms = tokenize(jl_path)
			for s in syms
				v = get!(h, s, 0)
				h[s] += 1
			end
		end
		h
	end
	reduce(merge_with!, r)
end
path = joinpath(DEPOT_PATH[1], "packages")
@elapsed h = distributed_histogram(path)
```

Second we try to distribute over all files.
```julia
function distributed_histogram_naive(path)
	jl_files = reduce(vcat, filter_jl(pkg_dir) for pkg_dir in sample_all_installed_pkgs(path))
	r = @showprogress pmap(jl_files) do jl_path
	# r = @showprogress pmap(WorkerPool([2,3,4,5]), jl_files) do jl_path
		h = Dict{Symbol, Int}()
		syms = tokenize(jl_path)
		for s in syms
			v = get!(h, s, 0)
			h[s] += 1
		end
		h
	end
	reduce(merge_with!, r)
end
path = joinpath(DEPOT_PATH[1], "packages")
@elapsed h = distributed_histogram_naive(path)
```

**BONUS**: You can do some analysis with `DataFrames`
```julia
using DataFrames
df = DataFrame(:sym => collect(keys(h)), :count => collect(values(h)));
sort!(df, :count, rev=true);
df[1:50,:]
```

```@raw html
</p></details>
```

## Threading

### Sum with threads

### Multithreaded file processing

## Task switching

### Only if we I come up with something interesting


## Summary
- when to use what

# Resources
- parallel computing [course](https://juliacomputing.com/resources/webinars/) by Julia Computing