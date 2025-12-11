# [Lab 10: Parallel computing](@id parallel_lab)
In this lab we are going to introduce tools that Julia's ecosystem offers for different ways of parallel computing. As an ilustration for how capable Julia was/is consider the fact that it has joined (alongside C,C++ and Fortran) the so-called "PetaFlop club"[^1], a list of languages capable of running at over 1PFLOPS.

[^1]: Blogpost "Julia Joins Petaflop Club" [https://juliacomputing.com/media/2017/09/julia-joins-petaflop-club/](https://juliacomputing.com/media/2017/09/julia-joins-petaflop-club/)

## Introduction
Nowadays there is no need to convince anyone about the advantages of having more cores available for your computation be it on a laptop, workstation or a cluster. The trend can be nicely illustrated in the figure bellow:
![42-cpu-trend](./42-years-processor-trend.png)
Image source[^2]

[^2]: Performance metrics trend of CPUs in the last 42years: [https://www.karlrupp.net/2018/02/42-years-of-microprocessor-trend-data/](https://www.karlrupp.net/2018/02/42-years-of-microprocessor-trend-data/)

However there are some shortcomings when going from sequential programming, that we have to note
- We don't think in parallel
- We learn to write and reason about programs serially
- The desire for parallelism often comes *after* you've written your algorithm (and found it too slow!)
- Harder to reason and therefore harder to debug
- The number of cores is increasing, thus knowing how the program scales is crucial (not just that it runs better)
- Benchmarking parallel code, that tries to exhaust the processor pool is much more affected by background processes

!!! warning "Shortcomings of parallelism"
    Parallel computing brings its own set of problems and not an insignificant overhead with data manipulation and communication, therefore try always to optimize your serial code as much as you can before advancing to parallel acceleration.

!!! warning "Disclaimer"
    With the increasing complexity of computer HW some statements may become outdated. Moreover we won't cover as many tips that you may encounter on a parallel programming specific course, which will teach you more in the direction of how to think in parallel, whereas here we will focus on the tools that you can use to realize the knowledge gained therein.

## Process based parallelism
As the name suggest process based parallelism is builds on the concept of running code on multiple processes, which can run even on multiple machines thus allowing to scale computing from a local machine to a whole network of machines - a major difference from the other parallel concept of threads. In Julia this concept is supported within standard library `Distributed` and the scaling to cluster can be realized by 3rd party library [`ClusterManagers.jl`](https://github.com/JuliaParallel/ClusterManagers.jl).

Let's start simply with knowing how to start up additional Julia processes. There are two ways:
- by adding processes using cmd line argument `-p ##`
```bash
julia -p 4
```
- by adding processes after startup using the `addprocs(##)` function from std library `Distributed`
```julia
julia> using Distributed
julia> addprocs(4) # returns a list of ids of individual processes
4-element Vector{Int64}:
 2
 3
 4
 5
julia> nworkers()  # returns number of workers
4
julia> nprocs()    # returns number of processes `nworkers() + 1`
5
```

The result shown in a process manager such as `htop`:
```bash
.../julia-1.6.2/bin/julia --project                                                                                     
.../julia-1.6.2/bin/julia -Cnative -J/home/honza/Apps/julia-1.6.2/lib/julia/sys.so -g1 --bind-to 127.0.0.1 --worker
.../julia-1.6.2/bin/julia -Cnative -J/home/honza/Apps/julia-1.6.2/lib/julia/sys.so -g1 --bind-to 127.0.0.1 --worker
.../julia-1.6.2/bin/julia -Cnative -J/home/honza/Apps/julia-1.6.2/lib/julia/sys.so -g1 --bind-to 127.0.0.1 --worker
.../julia-1.6.2/bin/julia -Cnative -J/home/honza/Apps/julia-1.6.2/lib/julia/sys.so -g1 --bind-to 127.0.0.1 --worker
```

Both of these result in total of 5 running processes - 1 controller, 4 workers - with their respective ids accessible via `myid()` function call. Note that the controller process has always id 1 and other processes are assigned subsequent integers, see for yourself with `@everywhere` macro, which runs easily code on all or a subset of processes.
```julia
@everywhere println(myid())
@everywhere [2,3] println(myid()) # select a subset of workers
```

The same way that we have added processes we can also remove them
```julia
julia> workers()   # returns array of worker ids
4-element Vector{Int64}:
 2
 3
 4
 5
julia> rmprocs(2)  # kills worker with id 2
Task (done) @0x00007ff2d66a5e40
julia> workers()
3-element Vector{Int64}:
 3
 4
 5
```

As we have seen from the `htop/top` output, added processes start with specific cmd line arguments, however they are not shared with any aliases that we may have defined, e.g. `julia` ~ `julia --project=.`. Therefore in order to use an environment, we have to first activate it on all processes
```julia
@everywhere begin
  using Pkg; Pkg.activate(@__DIR__) # @__DIR__ equivalent to a call to pwd()
end
```
or we can load files containing this line on all processes with cmdline option `-L ###.jl` together with `-p ##`.

There are generally two ways of working with multiple processes
- using low level functionality - we specify what/where is loaded, what/where is being run and when we fetch results
    + `@everywhere` to run everywhere and wait for completion
    + `@spawnat` and `remotecall` to run at specific process and return `Future` (a reference to a future result - remote reference)
    + `fetch` - fetching remote reference
    * `pmap` - for easily mapping a function over a collection

- using high level functionality - define only simple functions and apply them on collections
    + [`DistributedArrays`](https://github.com/JuliaParallel/DistributedArrays.jl)' with `DArray`s
    + [`Transducers.jl`](https://github.com/JuliaFolds/Transducers.jl) pipelines
    + [`Dagger.jl`](https://github.com/JuliaParallel/Dagger.jl) out-of-core and parallel computing

### Sum with processes
Writing your own sum of an array function is a good way to show all the potential problems, you may encounter with parallel programming. For comparison here is the naive version that uses `zero` for initialization and `@inbounds` for removing boundschecks.
```julia
function naive_sum(a)
    r = zero(eltype(a))
    for aᵢ in a
        r += aᵢ
    end
    r
end
```
Its performance will serve us as a sequential baseline.
```julia
julia> using BenchmarkTools
julia> a = rand(10_000_000); # 10^7
julia> sum(a) ≈ naive_sum(a)
true
julia> @btime sum($a)
5.011 ms (0 allocations: 0 bytes)
julia> @btime naive_sum($a)
11.786 ms (0 allocations: 0 bytes)
```
Note that the built-in `sum` exploits single core parallelism with Single instruction, multiple data (SIMD instructions) and is thus faster.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Write a distributed/multiprocessing version of `sum` function `dist_sum(a, np=nworkers())` without the help of `DistributedArrays`. Measure the speed up when doubling the number of workers (up to the number of logical cores - see [note](@ref lab10_thread) on hyper threading).

**HINTS**:
- map builtin `sum` over chunks of the array using `pmap`
- there are built in partition iterators `Iterators.partition(array, chunk_size)`
- `chunk_size` should relate to the number of available workers
- `pmap` has the option to pass the ids of workers as the second argument `pmap(f, WorkerPool([2,4]), collection)`
- `pmap` collects the partial results to the controller where it can be collected with another `sum`

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
```julia
using Distributed
addprocs(4)

@everywhere begin
    using Pkg; Pkg.activate(@__DIR__)
end

function dist_sum(a, np=nworkers())
    chunk_size = div(length(a), np)
    sum(pmap(sum, WorkerPool(workers()[1:np]), Iterators.partition(a, chunk_size)))
end

dist_sum(a) ≈ sum(a)
@btime dist_sum($a)

@time dist_sum(a, 1) # 74ms 
@time dist_sum(a, 2) # 46ms
@time dist_sum(a, 4) # 49ms
@time dist_sum(a, 8) # 35ms
```

```@raw html
</p></details>
```

As you can see the built-in `pmap` already abstracts quite a lot from the process and all the data movement is handled internally, however in order to show off how we can abstract even more, let's use the `DistributedArrays.jl` pkg.

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Write a distributed/multiprocessing version of `sum` function `dist_sum_lib(a, np=nworkers())` with the help of `DistributedArrays`. Measure the speed up when doubling the number of workers (up to the number of logical cores - see note on hyper threading).

**HINTS**:
- chunking and distributing the data can be handled for us using the `distribute` function on an array (creates a `DArray`)
- `distribute` has an option to specify on which workers should an array be distributed to
- `sum` function has a method for `DArray`
- remember to run `using DistributedArrays` on every process

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```
Setting up.
```julia
using Distributed
addprocs(8)

@everywhere begin
    using Pkg; Pkg.activate(@__DIR__)
end

@everywhere begin
    using DistributedArrays
end 
```

And the actual computation.
```julia
adist = distribute(a)       # distribute array to workers |> typeof - DArray
@time adist = distribute(a) # we should not disregard this time
@btime sum($adist)          # call the built-in function (dispatch on DArrray)

function dist_sum_lib(a, np=nworkers())
    adist = distribute(a, procs = workers()[1:np])
    sum(adist)
end

dist_sum_lib(a) ≈ sum(a)
@btime dist_sum_lib($a)

@time dist_sum_lib(a, 1) # 80ms 
@time dist_sum_lib(a, 2) # 54ms
@time dist_sum_lib(a, 4) # 48ms
@time dist_sum_lib(a, 8) # 33ms
```

```@raw html
</p></details>
```
In both previous examples we have included the data transfer time from the controller process, in practice however distributed computing is used in situations where the data may be stored on individual local machines. As a general rule of thumb we should always send only instruction what to do and not the actual data to be processed. This will be more clearly demonstrated in the next more practical example.

### [Distributed file processing](@id lab10_dist_file_p)
`Distributed` is often used in processing of files, such as the commonly encountered `mapreduce` jobs with technologies like [`Hadoop`](https://hadoop.apache.org/), [`Spark`](http://spark.apache.org/), where the files live on a distributed file system and a typical job requires us to map over all the files and gather some statistics such as histograms, sums and others. We will simulate this situation with the Julia's pkg codebase, which on a typical user installation can contain up to hundreds of thousand of `.jl` files (depending on how extensively one uses Julia).

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```

Write a distributed pipeline for computing a histogram of symbols found in AST by parsing Julia source files in your `.julia/packages/` directory. We have already implemented most of the code that you will need (available as source code [here](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/master/docs/src/lecture_10/pkg_processing.jl)).

```@raw html
<details>
<summary>pkg_processing.jl</summary><p>
```

```@example lab10_pkg_processing
using Markdown #hide
code = Markdown.parse("""```julia\n$(readchomp("./pkg_processing.jl"))\n```""") #hide
code
```

```@raw html
</p></details>
```


Your task is to write a function that does the `map` and `reduce` steps, that will create and gather the dictionaries from different workers. There are two ways to do a map
- either over directories inside `.julia/packages/` - call it `distributed_histogram_pkgwise`
- or over all files obtained by concatenation of `filter_jl` outputs (*NOTE* that this might not be possible if the listing itself is expensive - speed or memory requirements) - call it `distributed_histogram_filewise`
Measure if the speed up scales linearly with the number of processes by restricting the number of workers inside a `pmap`.

**HINTS**:
- for each file path apply `tokenize` to extract symbols and follow it with the update of a local histogram
- try writing sequential version first
- either load `./pkg_processing.jl` on startup with `-L` and `-p` options or `include("./pkg_processing.jl")` inside `@everywhere`
- use `pmap` to easily iterate in parallel over a collection - the result should be an array of histogram, which has to be merged on the controller node (use builtin `mergewith!` function in conjunction with `reduce`)
- `pmap` supports `do` syntax
```julia
pmap(collection) do item
    do_something(item)
end
```
- pkg directory can be obtained with `joinpath(DEPOT_PATH[1], "packages")`

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
@time h = sequential_histogram(path) # 87s
```

First we try to distribute over package folders. **TODO** add the ability to run it only on some workers
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

Merges count dictionary `h2` into `h1` by adding the counts. Equivalent to `Base.mergewith!(+)`.
"""
function merge_with!(h1, h2)
    for s in keys(h2)
        get!(h1, s, 0)
        h1[s] += h2[s]
    end
    h1
end

using ProgressMeter
function distributed_histogram_pkgwise(path, np=nworkers())
    r = @showprogress pmap(WorkerPool(workers()[1:np]), sample_all_installed_pkgs(path)) do pkg_dir
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

@time h = distributed_histogram_pkgwise(path, 2) # 41.5s
@time h = distributed_histogram_pkgwise(path, 4) # 24.0s
@time h = distributed_histogram_pkgwise(path, 8) # 24.0s
```

Second we try to distribute over all files.
```julia
function distributed_histogram_filewise(path, np=nworkers())
    jl_files = reduce(vcat, filter_jl(pkg_dir) for pkg_dir in sample_all_installed_pkgs(path))
    r = @showprogress pmap(WorkerPool(workers()[1:np]), jl_files) do jl_path
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
@time h = distributed_histogram_pkgwise(path, 2) # 46.9s
@time h = distributed_histogram_pkgwise(path, 4) # 24.8s
@time h = distributed_histogram_pkgwise(path, 8) # 20.4s
```
Here we can see that we have improved the timings a bit by increasing granularity of tasks.

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

## [Threading](@id lab10_thread)
The number of threads for a Julia process can be set up in an environmental variable `JULIA_NUM_THREADS` or directly on Julia startup with cmd line option `-t ##` or `--threads ##`. If both are specified the latter takes precedence.
```bash
julia -t 8
```
In order to find out how many threads are currently available, there exist the `nthreads` function inside `Base.Threads` library. There is also an analog to the Distributed `myid` example, called `threadid`.
```julia
julia> using Base.Threads
julia> nthreads()
8
julia> threadid()
1
```
As opposed to distributed/multiprocessing programming, threads have access to the whole memory of Julia's process, therefore we don't have to deal with separate environment manipulation, code loading and data transfers. However we have to be aware of the fact that memory can be modified from two different places and that there may be some performance penalties of accessing memory that is physically further from a given core (e.g. caches of different core or different NUMA[^3] nodes). Another significant difference from distributed computing is that we cannot spawn additional threads on the fly in the same way that we have been able to do with `addprocs` function.

[^3]: NUMA - [https://en.wikipedia.org/wiki/Non-uniform\_memory\_access](https://en.wikipedia.org/wiki/Non-uniform_memory_access)

!!! info "Hyper threads"
    In most of today's CPUs the number of threads is larger than the number of physical cores. These additional threads are usually called hyper threads[^4] or when talking about cores - logical cores. The technology relies on the fact, that for a given "instruction" there may be underutilized parts of the CPU core's machinery (such as one of many arithmetic units) and if a suitable work/instruction comes in it can be run simultaneously. In practice this means that adding more threads than physical cores may not be accompanied with the expected speed up.

[^4]: Hyperthreading - [https://en.wikipedia.org/wiki/Hyper-threading](https://en.wikipedia.org/wiki/Hyper-threading)

The easiest (not always yielding the correct result) way how to turn a code into multi threaded code is putting the `@threads` macro in front of a for loop, which instructs Julia to run the body on separate threads.
```julia
julia> A = Array{Union{Int,Missing}}(missing, nthreads());
julia> for i in 1:nthreads()
    A[threadid()] = threadid()
end
julia> A # only the first element is filled
8-element Vector{Union{Missing, Int64}}:
 1
  missing
  missing
  missing
  missing
  missing
  missing
  missing
```

```julia
julia> A = Array{Union{Int,Missing}}(missing, nthreads());
julia> @threads for i in 1:nthreads()
    A[threadid()] = threadid()
end
julia> A # the expected results
8-element Vector{Union{Missing, Int64}}:
 1
 2
 3
 4
 5
 6
 7
 8
```

### Multithreaded sum
Armed with this knowledge let's tackle the problem of the simple `sum`.
```julia
function threaded_sum_naive(a)
    r = zero(eltype(a))
    @threads for i in eachindex(a)
        @inbounds r += a[i]
    end
    return r
end
```
Comparing this with the built-in sum we see not an insignificant discrepancy (one that cannot be explained by reordering of computation) and moreover the timings show us some ridiculous overhead.
```julia
julia> using BenchmarkTools
julia> a = rand(10_000_000); # 10^7
julia> sum(a), threaded_sum_naive(a)
(5.000577175855193e6, 625888.2270955174)
julia> @btime sum($a)
  4.861 ms (0 allocations: 0 bytes)
julia> @btime threaded_sum_naive($a)
  163.379 ms (20000042 allocations: 305.18 MiB)
```
Recalling what has been said above we have to be aware of the fact that the data can be accessed from multiple threads at once, which if not taken into an account means that each thread reads possibly outdated value and overwrites it with its own updated state. 

There are two solutions which we will tackle in the next two exercises. 

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

**BONUS**:
Try chunking the array and calling sum on individual chunks to obtain some real speedup.

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

```julia
function threaded_sum_atom(a)
    r = Atomic{eltype(a)}(zero(eltype(a)))
    @threads for i in eachindex(a)
        @inbounds atomic_add!(r, a[i])
    end
    return r[]
end

julia> sum(a) ≈ threaded_sum_atom(a)
true
julia> @btime threaded_sum_atom($a)
  661.502 ms (42 allocations: 3.66 KiB)
```
That's better but far from the performance we need. 

**BONUS**: There is a fancier and faster way to do this by chunking the array
```julia
function threaded_sum_fancy_atom(a)
    r = Atomic{eltype(a)}(zero(eltype(a)))
    len, rem = divrem(length(a), nthreads())
    @threads for t in 1:nthreads()
        rₜ = zero(eltype(a))
        @simd for i in (1:len) .+ (t-1)*len
            @inbounds rₜ += a[i]
        end
        atomic_add!(r, rₜ)
    end
    # catch up any stragglers
    result = r[]
    @simd for i in length(a)-rem+1:length(a)
        @inbounds result += a[i]
    end
    return result
end

julia> sum(a) ≈ threaded_sum_fancy_atom(a)
true
julia> @btime threaded_sum_fancy_atom($a)
  2.983 ms (42 allocations: 3.67 KiB)
```

Finally we have beaten the "sequential" sum. The quotes are intentional, because the `Base`'s implementation of a sum uses Single instruction, multiple data (SIMD) instructions as well, which allow to process multiple elements at once.
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
function threaded_sum_buffer(a)
    R = zeros(eltype(a), nthreads())
    @threads for i in eachindex(a)
        @inbounds R[threadid()] += a[i]
    end
    r = zero(eltype(a))
    # sum the partial results from each thread
    for i in eachindex(R)
        @inbounds r += R[i]
    end
    return r
end

julia> sum(a) ≈ threaded_sum_buffer(a)
true
julia> @btime threaded_sum_buffer($a)
  2.750 ms (42 allocations: 3.78 KiB)
```

Though this implementation is cleaner and faster, there is possible drawback with this implementation, as the buffer `R` lives in a continuous part of the memory and each thread that accesses it brings it to its caches as a whole, thus invalidating the values for the other threads, which it in the same way.
```@raw html
</p></details>
```

Seeing how multithreading works on a simple example, let's apply it on the "more practical" case of the Symbol histogram from exercise [above](@ref lab10_dist_file_p).
### [Multithreaded file processing](@id lab10_dist_file_t)

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Exercise</header>
<div class="admonition-body">
```
Write a multithreaded analog of the file processing pipeline from [exercise](@ref lab10_dist_file_p) above. Again the task is to write the `map` and `reduce` steps, that will create and gather the dictionaries from different workers. There are two ways to map
- either over directories inside `.julia/packages/` - `threaded_histogram_pkgwise`
- or over all files obtained by concatenation of `filter_jl` outputs - `threaded_histogram_filewise`
Compare the speedup with the version using process based parallelism.

**HINTS**:
- create a separate dictionary for each thread in order to avoid the need for atomic operations
- 

**BONUS**:
In each of the cases count how many files/pkgs each thread processed. Would the dynamic scheduler help us in this situation?

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

Setup is now much simpler.
```julia
using Base.Threads
include("./pkg_processing.jl") 
path = joinpath(DEPOT_PATH[1], "packages")
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
    reduce(mergewith!(+), ht)
end

julia> @time h = threaded_histogram_pkgwise(path)
 26.958786 seconds (81.69 M allocations: 10.384 GiB, 4.58% gc time)
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
    reduce(mergewith!(+), ht)
end

julia> @time h = threaded_histogram_filewise(path)
 29.677184 seconds (81.66 M allocations: 10.411 GiB, 4.13% gc time)
```

```@raw html
</p></details>
```

## Task switching
There is a way how to run "multiple" things at once, which does not necessarily involve either threads or processes. In Julia this concept is called task switching or asynchronous programming, where we fire off our requests in a short time and let the cpu/os/network handle the distribution. As an example which we will try today is querying a web API, which has some variable latency. In the usuall sequantial fashion we can always post queries one at a time, however generally the APIs can handle multiple request at a time, therefore in order to better utilize them, we can call them asynchronously and fetch all results later, in some cases this will be faster.

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

How much time will the execution of each of them take?
```@raw html
<details class = "solution-body">
<summary class = "solution-header">Solution</summary><p>
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

# Resources
- parallel computing [course](https://juliacomputing.com/resources/webinars/) by Julia Computing