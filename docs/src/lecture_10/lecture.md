# Parallel programming with Julia
Julia offers different levels of parallel programming
- distributed processing, where jobs are split among different Julia processes
- multi-threadding, where jobs are split among multiple threads within the same processes
- SIMD instructions
- Task switching.

In this lecture, we will focus mainly on the first two, since SIMD instructions are mainly used for low-level optimization (such as writing you own very performant BLAS library), and task switching is not a true paralelism, but allows to run a different task when one task is waiting for example for IO.

## Process-level paralelism
Process-level paralelism means that Julia runs several compilers in different processes. By default, different processes *do not share anything by default*, meaning no libraries and variables. Everyhing has to be therefore set-up on all processes.

Julia off-the-shelf supports a mode, where a single *main* process controlls several workers. This main process has `myid() == 1`, worker processes receive higher numbers. Julia can be started with multiple workers from the very beggining, using `-p` switch as 
```julia
julia -p n
```
where `n` is the number of workers, or you can add workers after Julia has been started by
```julia
using Distributed
addprocs(4)
```
(when Julia is started with `-p`, `Distributed` library is loaded by default on main worker). You can also remove workers using `rmprocs`. Workers can be on the same physical machines, or on different machines. Julia offer integration via `ClusterManagers.jl` with most schedulling systems.

If you want evaluate piece of code on all workers including main process, a convenience macro `@everywhere` is offered.
```julia
@everywhere @show myid()
```
As we have mentioned, workers are loaded without libraries. We can see that by running
```julia
@everywhere InteractiveUtils.varinfo()
```
which fails, but
```julia
@everywhere begin 
	using InteractiveUtils
	println(InteractiveUtils.varinfo())
end
```

`@everywhere` macro allows us to define function and variables, and import libraries on workers as 
```julia
@everywhere begin 
	foo(x, y) = x * y + sin(y)
	foo(x) = foo(x, myid())
end
@everywhere @show foo(1.0)
```
Alternatively, we can put the code into a separate file and load it on all workers using `-L filename.jl`

Julia's multi-processing model is based on message-passing paradigm, but the abstraction is more akin to procedure calls. This means that users are saved from prepending messages with headers and implementing logic deciding which function should be called for thich header. Instead, we can *schedulle* an execution of a function on a remote worker and return the control immeadiately to continue in our job. A low-level function providing this functionality is `remotecall(fun, worker_id, args...)`. For example 
```julia
@everywhere begin 
	function delayed_foo(x, y, n )
		sleep(n)
		foo(x, y)
	end
end
r = remotecall(delayed_foo, 2, 1, 1, 60)
```
returns immediately, even though the function will take at least 60 seconds. `r` does not contain result of `foo(1, 1)`, but a struct `Future`, which is a *remote reference* in Julia's terminology. It points data located on some machine, indicates, if they are available and allows to `fetch` them from the remote worker. `fetch` is blocking, which means that the execution is blocked until data are available (if they are never available, the process can wait forever.) The presence of data can be checked using `isready`, which in case of `Future` returned from `remote_call` indicate that the computation has finished.
```julia
isready(r)
fetch(r) == foo(1, 1)
```
An advantage of the remote reference is that it can be freely shared around processes and the result can be retrieved on different node then the one which issued the call.s
```julia
r = remotecall(delayed_foo, 2, 1, 1, 60)
remotecall(r -> println("value: ",fetch(r), " retrieved on ", myid()) , 3, r)
```
An interesting feature of `fetch` is that it re-throw an exception raised on a different process.
```julia
@everywhere begin 
	function exfoo()
		throw("Exception from $(myid())")
	end
end
r = @spawnat 2 exfoo()
```
where `@spawnat` is a an alternative to `remotecall`, which executes a closure around expression (in this case `exfoo()`) on a specified worker (in this case 2). Fetching the result `r` throws an exception on the main process.
```julia
fetch(r)
```
`@spawnat` can be executed with `:any`  to signal that the user does not care, where the function will be executed and it will be left up to Julia.
```julia
r = @spawnat :any foo(1,1)
fetch(r)
```
Finally, if you would for some reason need to wait for the computed value, you can use 
```julia
remotecall_fetch(foo, 2, 1, 1)
```

## Example: Julia sets
Our example for explaining mechanisms of distributed computing will be the computation of Julia set fractal. The computation of the fractal can be easily paralelized, since the value of each pixel is independent from the remaining. The example is adapted from [Eric Aubanel](http://www.cs.unb.ca/~aubanel/JuliaMultithreadingNotes.html).
```julia
using Plots
@everywhere begin 
	function juliaset_pixel(z₀, c)
	    z = z₀
	    for i in 1:255
	        abs2(z)> 4.0 && return (i - 1)%UInt8
	        z = z*z + c
	    end
	    return UInt8(255)
	end

	function juliaset_column!(img, c, n, colj, j)
	    x = -2.0 + (j-1)*4.0/(n-1)
	    for i in 1:n
	        y = -2.0 + (i-1)*4.0/(n-1)
	        @inbounds img[i,colj] = juliaset_pixel(x+im*y, c)
	    end
	    nothing
	end
end

function juliaset(x, y, n=1000)
    c = x + y*im
    img = Array{UInt8,2}(undef,n,n)
    for j in 1:n
        juliaset_column!(img, c, n, j, j)
    end
    return img
end

frac = juliaset(-0.79, 0.15)
plot(heatmap(1:size(frac,1),1:size(frac,2), frac, color=:Spectral))
```

We can split the computation of the whole image into bands, such that each worker computes a smaller portion.
```julia
@everywhere begin 
	function juliaset_columns(c, n, columns)
	    img = Array{UInt8,2}(undef, n, length(columns))
	    for (colj, j) in enumerate(columns)
	        juliaset_column!(img, c, n, colj, j)
	    end
	    img
	end
end

function juliaset_spawn(x, y, n = 1000)
    c = x + y*im
    columns = Iterators.partition(1:n, div(n, nworkers()))
    r_bands = [@spawnat w juliaset_columns(c, n, cols) for (w, cols) in enumerate(columns)]
    slices = map(fetch, r_bands)
    reduce(hcat, slices)
end
```
we observe some speed-up over the serial version, but not linear in terms of number of workers
```julia
julia> @btime juliaset(-0.79, 0.15);
  38.699 ms (2 allocations: 976.70 KiB)

julia> @btime juliaset_spawn(-0.79, 0.15);
  21.521 ms (480 allocations: 1.93 MiB)
```
In the above example, we spawn one function on each worker and collect the results. In essence, we are performing `map` over bands. Julia offers for this usecase a parallel version of map `pmap`. With that, our example can look like
```julia
function juliaset_pmap(x, y, n = 1000, np = nworkers())
    c = x + y*im
    columns = Iterators.partition(1:n, div(n, np))
    slices = pmap(cols -> juliaset_columns(c, n, cols), columns)
    reduce(hcat, slices)
end

julia> @btime juliaset_pmap(-0.79, 0.15);
  17.597 ms (451 allocations: 1.93 MiB)
```
which has slightly better timing then the version based on `@spawnat` and `fetch` (as explained below in section about `Threads`, the parallel computation of Julia set suffers from each pixel taking different time to compute, which can be relieved by dividing the work into more parts --- `@btime juliaset_pmap(-0.79, 0.15, 1000, 16);`).

## Synchronization / Communication primitives
The orchestration of a complicated computation might be difficult with relatively low-level remote calls. A Producer / Consumer paradigm is a synchronization paradigm that uses queues. Consumer fetches work intructions from the queue and pushes results to different queue. Julia supports this paradigm with `Channel` and `RemoteChannel` primitives. Importantly, putting to and taking from queue is an atomic operation, hence we do not have take care of race conditions.
The code for the worker might look like
```julia
@everywhere begin 
	function juliaset_channel_worker(instructions, results)
		while isready(instructions)
			c, n, cols = take!(instructions)
			put!(results, (cols, juliaset_columns(c, n, cols)))
		end
	end
	println("finishing juliaset_channel_worker on ", myid())
end
```
The code for the main will look like
```julia
function juliaset_channels(x, y, n = 1000, np = nworkers())
	c = x + y*im
	columns = Iterators.partition(1:n, div(n, np))
	instructions = RemoteChannel(() -> Channel{Tuple}(np))
	foreach(cols -> put!(instructions, (c, n, cols)), columns)
	results = RemoteChannel(()->Channel{Tuple}(np))
	rfuns = [@spawnat i juliaset_channel_worker(instructions, results) for i in workers()]

	img = Array{UInt8,2}(undef, n, n)
	while isready(results)
		cols, impart = take!(results)
		img[:,cols] .= impart;
	end
	img
end

julia> @btime juliaset_channels(-0.79, 0.15);
  254.151 μs (254 allocations: 987.09 KiB)
```
The execution timw is much higher then what we have observed in the previous cases and changing the number of workers does not help much. What went wrong? The reason is that setting up the infrastructure around remote channels is a costly process. Consider the following alternative, where (i) we let workers to run endlessly and (ii) the channel infrastructure is set-up once and wrapped into an anonymous function
```julia
@everywhere begin 
	function juliaset_channel_worker(instructions, results)
		while true
		  c, n, cols = take!(instructions)
		  put!(results, (cols, juliaset_columns(c, n, cols)))
		end
	end
end

function juliaset_init(x, y, n = 1000, np = nworkers())
  c = x + y*im
  columns = Iterators.partition(1:n, div(n, np))
  T = Tuple{ComplexF64,Int64,UnitRange{Int64}}
  instructions = RemoteChannel(() -> Channel{T}(np))
  T = Tuple{UnitRange{Int64},Array{UInt8,2}}
  results = RemoteChannel(()->Channel{T}(np))
  foreach(p -> remote_do(juliaset_channel_worker, p, instructions, results), workers())
  function compute()
    img = Array{UInt8,2}(undef, n, n)
    foreach(cols -> put!(instructions, (c, n, cols)), columns)
    for i in 1:np
      cols, impart = take!(results)
      img[:,cols] .= impart;
    end
    img
  end 
end

t = juliaset_init(-0.79, 0.15)
julia> @btime t();
  17.697 ms (776 allocations: 1.94 MiB)
```
with which we obtain the comparable speed to the `pmap` approach.
!!! info
    ### `remote_do` vs `remote_call`
    Instead of `@spawnat` (`remote_call`) we can also use `remote_do` as foreach`(p -> remote_do(juliaset_channel_worker, p, instructions, results), workers)`, which executes the function `juliaset_channel_worker` at worker `p` with parameters `instructions` and `results` but does not return `Future` handle to receive the future results.

!!! info
    ### `Channel` and `RemoteChannel`
    `AbstractChannel` has to implement the interface `put!`, `take!`, `fetch`, `isready` and `wait`, i.e. it should behave like a queue. `Channel` is an implementation if an `AbstractChannel` that facilitates a communication within a single process (for the purpose of multi-threadding and task switching). Channel can be easily created by `Channel{T}(capacity)`, which can be infinite. The storage of a channel can be seen in `data` field, but a direct access will of course break all guarantees like atomicity of `take!` and `put!`.  For communication between proccesses, the `<:AbstractChannel` has to be wrapped in `RemoteChannel`. The constructor for `RemoteChannel(f::Function, pid::Integer=myid())` has a first argument a function (without arguments) which constructs the `Channel` (or something like that) on the remote machine identified by `pid` and returns the `RemoteChannel`. The storage thus resides on the machine specified by `pid` and the handle provided by the `RemoteChannel` can be freely passed to any process. (For curious, `ProcessGroup` `Distributed.PGRP` contains an information about channels on machines.) 

In the above example, `juliaset_channel_worker` defined as 
```julia
function juliaset_channel_worker(instructions, results)
	while true
	  c, n, cols = take!(instructions)
	  put!(results, (cols, juliaset_columns(c, n, cols)))
	end
end
```
runs forever due to the `while true` loop. To stop the computation, we usually extend the type accepted by the `instructions` channel to accept some stopping token (e.g. :stop) and stop.
```julia
function juliaset_channel_worker(instructions, results)
	while true
		i = take!(instructions)
		i === :stop && break
		c, n, cols = i
		put!(results, (cols, juliaset_columns(c, n, cols)))
	end
	put!(results, :stop)
end
```
Julia does not provide by default any facility to kill the remote execution except sending `ctrl-c` to the remote worker as `interrupt(pids::Integer...)`.

## Sending data
Sending parameters of functions and receiving results from a remotely called functions migh incur a significant cost. 
1. Try to minimize the data movement as much as possible. A prototypical example is
```julia
A = rand(1000,1000);
Bref = @spawnat :any A^2;
```
and
```julia
Bref = @spawnat :any rand(1000,1000)^2;
```
2. It is not only volume of data (in terms of the number of bytes), but also a complexity of objects that are being sent. Serialization can be very time consuming, an efficient converstion to something simple might be wort
```julia
using BenchmarkTools
@everywhere begin 
	using Random
	v = [randstring(rand(1:20)) for i in 1:1000];
	p = [i => v[i] for i in 1:1000]
	d = Dict(p)

	send_vec() = v
	send_dict() = d
	send_pairs() = p
	custom_serialization() = (length.(v), join(v, ""))
end

@btime remotecall_fetch(send_vec, 2);
@btime remotecall_fetch(send_dict, 2);
@btime remotecall_fetch(send_pairs, 2);
@btime remotecall_fetch(custom_serialization, 2);
```
3. Some type of objects cannot be properly serialized and deserialized
```julia
a = IdDict(
	:a => rand(1,1),
	)
b = remotecall_fetch(identity, 2, a)
a[:a] === a[:a]
a[:a] === b[:a]
```
4. If you need to send the data to worker, i.e. you want to define (overwrite) a global variable there
```julia
@everywhere begin 
	g = rand()
	show_secret() = println("secret of ", myid(), " is ", g)
end
@everywhere show_secret()

remotecall_fetch(g -> eval(:(g = $(g))), 2, g)
@everywhere show_secret()
```
which is implemented in the 

## Practical advices 
Recall that (i) workers are started as clean processes and (ii) they might not share the same environment with the main process. The latter is due to the fact that files describing the environment (`Project.toml` and `Manifest.toml`) might not be available on remote machines.
We recommend:
- to have shared directory (shared home) with code and to share the location of packages
- to place all code for workers to one file, let's call it `worker.jl` (author of this includes the code for master as well).
- put to the beggining of `worker.jl` code activating specified environment as 
```julia
using Pkg
Pkg.activate(@__DIR__)
```
and optionally
```julia
Pkg.resolve()
Pkg.instantiate()
```
- run julia as
```julia
julia -p ?? -L worker.jl main.jl
```
where `main.jl` is the script to be executed on the main node. Or
```julia
julia -p ?? -L worker.jl -e "main()"
```
where `main()` is the function defined in `worker.jl` to be executed on the main node.

A complete example can be seen in [`juliaset_p.jl`](juliaset_p.jl).


## Multi-Threadding 
- Locks / lock-free multi-threadding
- Show the effect of different schedullers
- intra-model parallelism
- sucks when operating with Heap

## Julia sets
An example adapted from [Eric Aubanel](http://www.cs.unb.ca/~aubanel/JuliaMultithreadingNotes.html).

For ilustration, we will use Julia set fractals, ad they can be easily paralelized. Some fractals (Julia set, Mandelbrot) are determined by properties of some complex-valued functions. Julia set counts, how many iteration is required for  ``f(z) = z^2+c`` to be bigger than two in absolute value, ``|f(z)| >=2 ``. The number of iterations can then be mapped to the pixel's color, which creates a nice visualization we know.
```julia
function juliaset_pixel(z₀, c)
    z = z₀
    for i in 1:255
        abs2(z)> 4.0 && return (i - 1)%UInt8
        z = z*z + c
    end
    return UInt8(255)
end
```

In our multi-threadding experiments, the level of granulity will be one column, since calculation of single pixel is so fast, that thread-switching will have much higher overhead.
```julia
function juliaset_column!(img, c, n, j)
    x = -2.0 + (j-1)*4.0/(n-1)
    for i in 1:n
        y = -2.0 + (i-1)*4.0/(n-1)
        @inbounds img[i,j] = juliaset_pixel(x+im*y, c)
    end
    nothing
end
```

To calculate full image
```julia
function juliaset_image!(img, c, n)
    for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end
```

```julia
function juliaset(x, y, n=1000, method = juliaset_image!, extra...)
    c = x + y*im
    img = Array{UInt8,2}(undef,n,n)
    method(img, c, n, extra...)
    return img
end
```

and run it
```julia
using Plots
frac = juliaset(-0.79, 0.15)
plot(heatmap(1:size(frac,1),1:size(frac,2), frac, color=:Spectral))
```

To observe the execution length, we will use `BenchmarkTools.jl` again
```
using BenchmarkTools
julia> @btime juliaset(-0.79, 0.15);
  39.822 ms (2 allocations: 976.70 KiB)
```

Let's now try to speed-up the calculation using multi-threadding. `Julia v0.5` has introduced multi-threadding with static-scheduller with a simple syntax: just prepend the for-loop with a `Threads.@threads` macro. With that, the first multi-threaded version will looks like
```julia
function juliaset_static!(img, c, n)
    Threads.@threads for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end
```
with benchmark
```
julia> @btime juliaset(-0.79, 0.15, 1000, juliaset_static!);
  16.206 ms (26 allocations: 978.42 KiB)
```
Although we have used four-threads, the speed improvement is ``2.4``. Why is that? The reason is that the static scheduller partition the total number of columns (1000) into equal parts, where the total number of parts is equal to the number of threads, and assign each to a single thread. In our case, we will have four parts each of size 250. Since the execution time of computing value of each pixel is not the same and therefore there will be threads who will not do anything. This problem is addressed by dynamic schedulling, which divides the problem into smaller parts, and when a thread is finished with one part, it assigned new not-yet computed part.

Dynamic scheduller is supported using `Threads.@spawn` macro. The prototypical approach is the fork-join model, where one recursivelly partitions the problems and wait in each thread for the other
```julia
function juliaset_recspawn!(img, c, n, lo=1, hi=n, ntasks=128)
    if hi - lo > n/ntasks-1
        mid = (lo+hi)>>>1
        finish = Threads.@spawn juliaset_recspawn!(img, c, n, lo, mid, ntasks)
        juliaset_recspawn!(img, c, n, mid+1, hi, ntasks)
        wait(finish)
        return
    end
    for j in lo:hi
        juliaset_column!(img, c, n, j)
    end
    nothing
end
```
Measuring the time we observe four-times speedup, which corresponds to the number of threads.
```julia
julia> @btime juliaset(-0.79, 0.15, 1000, juliaset_recspawn!);
  10.326 ms (142 allocations: 986.83 KiB)
```
Due to task switching overhead, increasing the granularity might not pay off.
```julia
4 tasks: 16.262 ms (21 allocations: 978.05 KiB)
8 tasks: 10.660 ms (45 allocations: 979.80 KiB)
16 tasks: 10.326 ms (142 allocations: 986.83 KiB)
32 tasks: 10.786 ms (238 allocations: 993.83 KiB)
64 tasks: 10.211 ms (624 allocations: 1021.89 KiB)
128 tasks: 10.224 ms (1391 allocations: 1.05 MiB)
256 tasks: 10.617 ms (2927 allocations: 1.16 MiB)
512 tasks: 11.012 ms (5999 allocations: 1.38 MiB)
```

```julia
using FLoops, FoldsThreads
function juliaset_folds!(img, c, n)
    @floop ThreadedEx(basesize = 2) for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end
julia> @btime juliaset(-0.79, 0.15, 1000, juliaset_folds!);
  10.575 ms (52 allocations: 980.12 KiB)
```
where `basesize` is the size of the part, in this case 2 columns.
```julia
function juliaset_folds!(img, c, n)
    @floop WorkStealingEx(basesize = 2) for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end
julia> @btime juliaset(-0.79, 0.15, 1000, juliaset_folds!);
  10.575 ms (52 allocations: 980.12 KiB)
```

```julia
function juliaset_folds!(img, c, n)
    @floop DepthFirstEx(basesize = 2) for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    nothing
end
julia> @btime juliaset(-0.79, 0.15, 1000, juliaset_folds!);
  10.421 ms (3582 allocations: 1.20 MiB)
```

## Take away message
When deciding, what kind of paralelism to employ, consider following
- for tightly coupled computation over shared data, multi-threadding is more suitable due to non-existing sharing of data between processes
- but if the computation requires frequent allocation and freeing of memery, or IO, separate processes are multi-suitable, since garbage collectors are independent between processes
- `Transducers` thrives for (almost) the same code to support thread- and process-based paralelism.

### Materials
- http://cecileane.github.io/computingtools/pages/notes1209.html
- https://lucris.lub.lu.se/ws/portalfiles/portal/61129522/julia_parallel.pdf
- https://www.csd.uwo.ca/~mmorenom/cs2101a_moreno/Parallel_computing_with_Julia.pdf
- Threads: https://juliahighperformance.com/code/Chapter09.html
- Processes: https://juliahighperformance.com/code/Chapter10.html
- Alan Adelman uses FLoops in https://www.youtube.com/watch?v=dczkYlOM2sg
- Examples: ?Heat equation? from https://hpc.llnl.gov/training/tutorials/introduction-parallel-computing-tutorial#Examples