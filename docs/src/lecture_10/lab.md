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
The number of threads that Julia can use can be set up in an environmental variable `JULIA_NUM_THREADS` or directly on julia startup with cmd line option `-t ##` or `--threads ##`. If both are specified the latter takes precedence.
```bash
julia -t 8
```
In order to find out how many threads are currently available, there exist the `nthreads` function inside `Base.Threads` library. There is also an analog to the Distributed `myid` example, called `threadid`.
```julia
using Base.Threads
nthreads()
threadid()
```
As opposed to distributed/multiprocessing programming, threads have access to the whole memory of Julia's process, therefore we don't have to deal with separate environment manipulation, code loading and data transfers. However we have to be aware of the fact that memory can be modified from two different places and that there may be some performance penalties of accessing memory that is physically further from a given core (e.g. caches of different core or different NUMA[^2] nodes)

[^2]: NUMA - [https://en.wikipedia.org/wiki/Non-uniform\_memory\_access](https://en.wikipedia.org/wiki/Non-uniform_memory_access)

!!! info "Hyper threads"
    In most of today's CPUs the number of threads is larger than the number of physical cores. These additional threads are usually called hyper threads[^3]. The technology relies on the fact, that for a given "instruction" there may be underutilized parts of the CPU core's machinery (such as one of many arithmetic units) and if a suitable work/instruction comes in it can be run simultaneously. In practice this means that adding more threads than physical cores may not accompanied with the expected speed up.

[^3]: Hyperthreading - [https://en.wikipedia.org/wiki/Hyper-threading](https://en.wikipedia.org/wiki/Hyper-threading)

The easiest (not always yielding the correct result) way how to turn a code into multi threaded code is putting the `@threads` macro in front of a for loop, which instructs Julia to run the body on separate threads.
```julia
A = Array{Union{Int,Missing}}(missing, nthreads())
for i in 1:nthreads()
    A[threadid()] = threadid()
end
A # only the first element is filled
```

```julia
@threads for i in 1:nthreads()
    A[threadid()] = threadid()
end
A # the expected results
```

### Multithreaded sum
Armed with this knowledge let's tackle the problem of a simple sum.
```julia
function threaded_sum_naive(A)
    r = zero(eltype(A))
    @threads for i in eachindex(A)
        @inbounds r += A[i]
    end
    return r
end
```
Comparing this with the built-in sum we see not an insignificant discrepancy (one that cannot be explained by reordering of computation)
```julia
a = rand(10_000_000);
sum(a), threaded_sum_naive(a)
```
Recalling what has been said above we have to be aware of the fact that the data can be accessed from multiple threads at once, which if not taken into an account means that each thread reads possibly outdated value and overwrites it with its own updated state. There are two solutions which we will tackle in the next two exercises.


```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement `threaded_sum_atom`, which uses `Atomic` wrapper around the accumulator variable `r` in order to ensure correct locking of data access. 

**HINTS**:
- use `atomic_add!` as a replacement of `r += A[i]`
- "collect" the result by dereferencing variable `r` with empty bracket operator `[]`

!!! info "Side note on dereferencing"
    In Julia we can create references to a data types, which are guarranteed to point to correct and allocated type in memory, as long as a reference exists the memory is not garbage collected. These are constructed with `Ref(x)`, `Ref(a, 7)` or `Ref{T}()` for reference to variable `x`, `7`th element of array `a` and an empty reference respectively. Dereferencing aka asking about the underlying value is done using empty bracket operator `[]`.
    ```@repl lab10_refs
    x = 1       # integer
    rx = Ref(x) # reference to that particular integer `x`
    x == rx[]   # dereferencing yields the same value
    ```
    There also exist unsafe references/pointers `Ptr`, however we should not really come into a contact with those.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
using BenchmarkTools
a = rand(10^7);

function threaded_sum_atom(A)
    r = Atomic{eltype(A)}(zero(eltype(A)))
    @threads for i in eachindex(A)
        @inbounds atomic_add!(r, A[i])
    end
    return r[]
end
```
There is a fancier and faster way to do this by chunking the array, because this is comparable in speed to sequential code.

```julia
function threaded_sum_fancy_atomic(A)
    r = Atomic{eltype(A)}(zero(eltype(A)))
    len, rem = divrem(length(A), nthreads())
    @threads for t in 1:nthreads()
        rₜ = zero(eltype(A))
        @simd for i in (1:len) .+ (t-1)*len
            @inbounds rₜ += A[i]
        end
        atomic_add!(r, rₜ)
    end
    # catch up any stragglers
    result = r[]
    @simd for i in length(A)-rem+1:length(A)
        @inbounds result += A[i]
    end
    return result
end
```

```@raw html
</p></details>
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Implement `threaded_sum_buffer`, which uses an array of length `nthreads()` (we will call this buffer) for local aggregation of results of individual threads. 

**HINTS**:
- use `threadid()` to index the buffer array
- sum the buffer array to obtain final result


```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
using BenchmarkTools
a = rand(10^7);

function threaded_sum_buffer(A)
    R = zeros(eltype(A), nthreads())
    @threads for i in eachindex(A)
        @inbounds R[threadid()] += A[i]
    end
    r = zero(eltype(A))
    # sum the partial results from each thread
    for i in eachindex(R)
        @inbounds r += R[i]
    end
    return r
end
```

```@raw html
</p></details>
```

### Multithreaded file processing

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Write a multithreaded analog of the file processing pipeline from [exercise](@ref lab10_dist_file_p) above. We have already implemented most of the code that you will need (available as source code [here](source code)). **TODO**

Your job is to write the `map` and `reduce` steps, that will gather the dictionaries from different workers. There are two ways to map
- either over directories inside `.julia/packages/`
- or over all files obtained by concatenation of `filter_jl` outputs (*NOTE* that this might not be possible if the listing itself is expensive - speed or memory requirements)
Measure if the speed up scales linearly with the number of threads in each case. (*NOTE* that this )


**HINTS**:
- create a separate dictionary for each thread in order to avoid the need for atomic operations


**BONUS**:
In each of the cases count how many files/pkgs each thread processed. Would the dynamic scheduler help us in this situation?

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
using Base.Threads
include("./pkg_processing.jl") 

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
```

Firstly the version with folder-wise parallelism.
```julia
function threaded_histogram_pkgwise(path)
    ht = [Dict{Symbol, Int}() for _ in 1:nthreads()]
    @threads for pkg_dir in sample_all_installed_pkgs(path)
        h = ht[threadid()]
        for jl_path in filter_jl(pkg_dir)
            syms = tokenize(jl_path)
            for s in syms
                v = get!(h, s, 0)
                h[s] += 1
            end
        end
    end
    reduce(merge_with!, ht)
end
path = joinpath(DEPOT_PATH[1], "packages")
@time h = threaded_histogram_pkgwise(path)
```

Secondly the version with file-wise parallelism.
```julia
function threaded_histogram_filewise(path)
    jl_files = reduce(vcat, filter_jl(pkg_dir) for pkg_dir in sample_all_installed_pkgs(path))
    ht = [Dict{Symbol, Int}() for _ in 1:nthreads()]
    @threads for jl_path in jl_files
        h = ht[threadid()]
        syms = tokenize(jl_path)
        for s in syms
            v = get!(h, s, 0)
            h[s] += 1
        end
    end
    reduce(merge_with!, ht)
end
path = joinpath(DEPOT_PATH[1], "packages")
@time h = threaded_histogram_filewise(path)
```

```@raw html
</p></details>
```

## Task switching
There is a way how to run "multiple" things at once, which does not necessarily involve either threads or processes. In Julia this concept is called task switching or asynchronous programming, where we fire off our requests in a short time and let the cpu/os/network handle the distribution. As an example which we will try today is querrying a web API, which has some variable latency. In the usuall sequantial fashion we can always post querries one at a time, however generally the APIs can handle multiple request at a time, therefore in order to better utilize them, we can call them asynchronously and fetch all results later, in some cases this will be faster.

!!! info "Burst requests"
    It is a good practice to check if an API supports some sort of batch request, because making a burst of single request might lead to a worse performance for others and a possible blocking of your IP/API key.

Consider following functions
```julia
function a()
    for i in 1:10
        sleep(1)
    end
end

function b()
    for i in 1:10
        @async sleep(1)
    end
end

function c()
    @sync for i in 1:10
        @async sleep(1)
    end
end
```

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">How much time will take the execution of each of them?:</summary><p>
```
```
```julia
@time a() # 10s
@time b() # ~0s
@time c() # >~1s
```
```@raw html
</p></details>
```

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Choose one of the free web APIs and query its endpoint using the `HTTP.jl` library. Implement both sequential and asynchronous version. Compare them on an burst of 10 requests.

**HINTS**:
- use `HTTP.request` for `GET` requests on your chosen API, e.g. `r = HTTP.request("GET", "https://catfact.ninja/fact")` for random cat fact
- converting body of a response can be done simply by constructing a `String` out of it - `String(r.body)`
- in order to parse a json string use `JSON.jl`'s parse function
- Julia offers `asyncmap` - asynchronous `map`

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
using HTTP, JSON

function query_cat_fact()
    r = HTTP.request("GET", "https://catfact.ninja/fact")
    j = String(r.body)
    d = JSON.parse(j)
    d["fact"]
end

# without asyncmap
function get_cat_facts_async(n)
    facts = Vector{String}(undef, n)
    @sync for i in 1:10
        @async facts[i] = query_cat_fact()
    end
    facts
end

get_cat_facts_async(n) = asyncmap(x -> query_cat_fact(), Base.OneTo(n))
get_cat_facts(n) = map(x -> query_cat_fact(), Base.OneTo(n))

@time get_cat_facts_async(10)   # ~0.15s
@time get_cat_facts(10)         # ~1.1s
```

```@raw html
</p></details>
```


## Summary
- when to use what

# Resources
- parallel computing [course](https://juliacomputing.com/resources/webinars/) by Julia Computing