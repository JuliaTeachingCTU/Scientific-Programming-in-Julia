# Parallel programming with Julia
Julia offers different levels of parallel programming
- distributed processing, where jobs are split among different Julia processes
- multi-threadding, where jobs are split among multiple threads within the same processes
- SIMD instructions
- Task switching.

In this lecture, we will focus mainly on the first two, since SIMD instructions are mainly used for low-level optimization (such as writing your own very performant BLAS library), and task switching is not a true paralelism, but allows to run a different task when one task is waiting for example for IO.

**The most important lesson is that before you jump into the parallelism, be certain you have made your sequential code as fast as possible.**

## Process-level paralelism
Process-level paralelism means we run several instances of Julia (in different processes) and they communicate between each other using inter-process communication (IPC). The implementation of IPC differs if parallel julia instances share the same machine, or they are on different machines spread over the network. By default, different processes *do not share any libraries or any variables*. They are loaded clean and it is up to the user to set-up all needed code and data.

Julia's default modus operandi is a single *main* instance controlling several workers. This main instance has `myid() == 1`, worker processes receive higher numbers. Julia can be started with multiple workers from the very beggining, using `-p` switch as 
```julia
julia -p n
```
where `n` is the number of workers, or you can add workers after Julia has been started by
```julia
using Distributed
addprocs(n)
```
You can also remove workers using `rmprocs`.  When Julia is started with `-p`, `Distributed` library is loaded by default on main worker. Workers can be on the same physical machines, or on different machines. Julia offer integration via `ClusterManagers.jl` with most schedulling systems.

If you want to evaluate piece of code on all workers including main process, a convenience macro `@everywhere` is offered.
```julia
@everywhere @show myid()
```
As we have mentioned, workers are loaded without libraries. We can see that by running
```julia
@everywhere InteractiveUtils.varinfo()
```
which fails, but after loading `InteractiveUtils` everywhere
```julia
using Statistics
@everywhere begin 
	using InteractiveUtils
	println(InteractiveUtils.varinfo(;imported = true))
end
```
we see that `Statistics` was loaded only on the main process. Thus, there is not magical sharing of data and code.
With `@everywhere` macro we can define function and variables, and import libraries on workers as 
```julia
@everywhere begin 
	foo(x, y) = x * y + sin(y)
	foo(x) = foo(x, myid())
	x = rand()
end
@everywhere @show foo(1.0)
@everywhere @show x
```
The fact that `x` has different values on different workers and master demonstrates again the independency of processes. While we can set up everything using `@everywhere` macro, we can also put all the code for workers into a separate file, e.g. `worker.jl` and load it on all workers using `-L worker.jl`.

Julia's multi-processing model is based on message-passing paradigm, but the abstraction is more akin to procedure calls. This means that users are saved from prepending messages with headers and implementing logic deciding which function should be called for thich header. Instead, we can *schedulle* an execution of a function on a remote worker and return the control immeadiately to continue in our job. A low-level function providing this functionality is `remotecall(fun, worker_id, args...)`. For example 
```julia
@everywhere begin 
	function delayed_foo(x, y, n )
		sleep(n)
		println("woked up")
		foo(x, y)
	end
end
r = remotecall(delayed_foo, 2, 1, 1, 60)
```
returns immediately, even though the function will take at least 60 seconds. `r` does not contain result of `foo(1, 1)`, but a struct `Future`, which is a *remote reference* in Julia's terminology. It points to data located on some machine, indicates, if they are available and allows to `fetch` them from the remote worker. `fetch` is blocking, which means that the execution is blocked until data are available (if they are never available, the process can wait forever.) The presence of data can be checked using `isready`, which in case of `Future` returned from `remote_call` indicate that the computation has finished.
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
where we have used `@spawnat` instead of `remote_call`. It is higher level alternative executing a closure around the expression (in this case `exfoo()`) on a specified worker, in this case 2. Coming back to the example, when we fetch the result `r`, the exception is throwed on the main process, not on the worker
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

## Running example: Julia sets
Our example for explaining mechanisms of distributed computing will be Julia set fractals, as they can be easily paralelized.  The example is adapted from [Eric Aubanel](http://www.cs.unb.ca/~aubanel/JuliaMultithreadingNotes.html). Some fractals (Julia set, Mandelbrot) are determined by properties of some complex-valued functions. Julia set counts, how many iteration is required for  ``f(z) = z^2+c`` to be bigger than two in absolute value, ``|f(z)| >=2 ``. The number of iterations can then be mapped to the pixel's color, which creates a nice visualization we know.
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
A nice property of fractals like Julia set is that the computation can be easily paralelized, since the value of each pixel is independent from the remaining. In our experiments, the level of granulity will be one column, since calculation of single pixel is so fast, that thread or process switching will have much higher overhead.
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
function juliaset(x, y, n=1000)
    c = x + y*im
    img = Array{UInt8,2}(undef,n,n)
    for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    return img
end
```
and run it and view it
```julia
using Plots
frac = juliaset(-0.79, 0.15)
plot(heatmap(1:size(frac,1),1:size(frac,2), frac, color=:Spectral))
```
or with GLMakie
```julia
using GLMakie
frac = juliaset(-0.79, 0.15)
heatmap(frac)
```
To observe the execution length, we will use `BenchmarkTools.jl` 
```
using BenchmarkTools
julia> @btime juliaset(-0.79, 0.15);
  39.822 ms (2 allocations: 976.70 KiB)
```

Let's now try to speed-up the computation using more processes. We first make functions available to workers
```julia
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
```
For the actual parallelisation, we split the computation of the whole image into bands, such that each worker computes a smaller portion.
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
which has slightly better timing then the version based on `@spawnat` and `fetch` (as explained below in section about `Threads`, the parallel computation of Julia set suffers from each pixel taking different time to compute, which can be relieved by dividing the work into more parts:
```julia
julia> @btime juliaset_pmap(-0.79, 0.15, 1000, 16);
  12.686 ms (1439 allocations: 1.96 MiB)
```

## Shared memory
When main and all workers are located on the same process, and the OS supports sharing memory between processes (by sharing memory pages), we can use `SharedArrays` to avoid sending the matrix with results.
```julia
@everywhere begin
	using SharedArrays
	function juliaset_shared(x, y, n=1000)
	    c = x + y*im
	    img = SharedArray(Array{UInt8,2}(undef,n,n))
	    @sync @distributed for j in 1:n
	        juliaset_column!(img, c, n, j, j)
	    end
	    return img
	end 
end

julia> @btime juliaset_shared(-0.79, 0.15);
  19.088 ms (963 allocations: 1017.92 KiB)
```

The allocation of the Shared Array mich be costly, let's try to put the allocation outside of the loop
```julia
img = SharedArray(Array{UInt8,2}(undef,1000,1000))
function juliaset_shared!(img, x, y, n=1000)
    c = x + y*im
    @sync @distributed for j in 1:n
        juliaset_column!(img, c, n, j, j)
    end
    return img
end 

julia> @btime juliaset_shared!(img, -0.79, 0.15);
  17.399 ms (614 allocations: 27.61 KiB)
```
but both versions are not akin. It seems like the alocation of `SharedArray` costs approximately `2`ms.

`@distributed for` (`Distributed.pfor`) does not allows to supply, as it splits the for cycle to `nworkers()` processes. Above we have seen that more splits is better
```julia
@everywhere begin 
	function juliaset_columns!(img, c, n, columns)
	    for (colj, j) in enumerate(columns)
	        juliaset_column!(img, c, n, colj, j)
	    end
	end
end

img = SharedArray(Array{UInt8,2}(undef,1000,1000))
function juliaset_shared!(img, x, y, n=1000, np = nworkers())
    c = x + y*im
    columns = Iterators.partition(1:n, div(n, np))
    pmap(cols -> juliaset_columns!(img, c, n, cols), columns)
    return img
end 

julia> @btime juliaset_shared!(img, -0.79, 0.15, 1000, 16);
  11.760 ms (1710 allocations: 85.98 KiB)
```
Which is almost 1ms faster than without used of pre-allocated `SharedArray`. Notice the speedup is now `38.699 / 11.76 = 3.29×`

## Synchronization / Communication primitives
The orchestration of a complicated computation might be difficult with relatively low-level remote calls. A *producer / consumer* paradigm is a synchronization paradigm that uses queues. Consumer fetches work intructions from the queue and pushes results to different queue. Julia supports this paradigm with `Channel` and `RemoteChannel` primitives. Importantly, putting to and taking from queue is an atomic operation, hence we do not have take care of race conditions.
The code for the worker might look like
```julia
@everywhere begin 
	function juliaset_channel_worker(instructions, results)
		while isready(instructions)
			c, n, cols = take!(instructions)
			put!(results, (cols, juliaset_columns(c, n, cols)))
		end
	end
end
```
The code for the main will look like
```julia
function juliaset_channels(x, y, n = 1000, np = nworkers())
	c = x + y*im
	columns = Iterators.partition(1:n, div(n, np))
	instructions = RemoteChannel(() -> Channel(np))
	foreach(cols -> put!(instructions, (c, n, cols)), columns)
	results = RemoteChannel(()->Channel(np))
	rfuns = [@spawnat i juliaset_channel_worker(instructions, results) for i in workers()]

	img = Array{UInt8,2}(undef, n, n)
	for i in 1:np
		cols, impart = take!(results)
		img[:,cols] .= impart;
	end
	img
end

julia> @btime juliaset_channels(-0.79, 0.15);
```
The execution time is much higher then what we have observed in the previous cases and changing the number of workers does not help much. What went wrong? The reason is that setting up the infrastructure around remote channels is a costly process. Consider the following alternative, where (i) we let workers to run endlessly and (ii) the channel infrastructure is set-up once and wrapped into an anonymous function
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
runs forever due to the `while true` loop. 

Julia does not provide by default any facility to kill the remote execution except sending `ctrl-c` to the remote worker as `interrupt(pids::Integer...)`. To stop the computation, we usually extend the type accepted by the `instructions` channel to accept some stopping token (e.g. :stop) and stop.
```julia
@everywhere begin 
	function juliaset_channel_worker(instructions, results)
		while true
			i = take!(instructions)
			i === :stop && break
			c, n, cols = i
			put!(results, (cols, juliaset_columns(c, n, cols)))
		end
		println("worker $(myid()) stopped")
		put!(results, :stop)
	end
end

function juliaset_init(x, y, n = 1000, np = nworkers())
  c = x + y*im
  columns = Iterators.partition(1:n, div(n, np))
  instructions = RemoteChannel(() -> Channel(np))
  results = RemoteChannel(()->Channel(np))
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
t()
foreach(i -> put!(t.instructions, :stop), workers())
```
In the above example we paid the price of introducing type instability into the channels, which now contain types `Any` instead of carefully constructed tuples. But the impact on the overall running time is negligible
```julia
t = juliaset_init(-0.79, 0.15)
julia> @btime t()
  17.551 ms (774 allocations: 1.94 MiB)
foreach(i -> put!(t.instructions, :stop), workers())
```
In some use-cases, the alternative can be to put all jobs to the `RemoteChannel` before workers are started, and then stop the workers when the remote channel is empty as 
```julia
@everywhere begin 
	function juliaset_channel_worker(instructions, results)
		while !isready(instructions)
		  c, n, cols = take!(instructions)
		  put!(results, (cols, juliaset_columns(c, n, cols)))
		end
	end
end
``` 

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
2. It is not only volume of data (in terms of the number of bytes), but also a complexity of objects that are being sent. Serialization can be very time consuming, an efficient converstion to something simple might be worth
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

for i in workers()
	remotecall_fetch(g -> eval(:(g = $(g))), i, g)
end
@everywhere show_secret()
```
which is implemented in the `ParallelDataTransfer.jl` with other variants, but in general, this construct should be avoided.

Alternatively, you can overwrite a global variable
```julia
@everywhere begin 
	g = rand()
	show_secret() = println("secret of ", myid(), " is ", g)
	function set_g(x) 
		global g
		g = x
	end
end

@everywhere show_secret()
remote_do(set_g, 2, 2)
@everywhere show_secret()
```

## Practical advices 
Recall that (i) workers are started as clean processes and (ii) they might not share the same environment with the main process. The latter is due to the possibility of remote machines to have a different directory structure. 
```julia
@everywhere begin 
	using Pkg
	println(Pkg.project().path)
end
```
Our advices earned by practice are:
- to have shared directory (shared home) with code and to share the location of packages
- to place all code for workers to one file, let's call it `worker.jl` (author of this includes the code for master as well).
- put to the beggining of `worker.jl` code activating specified environment as (or specify environmnet for all workers in environment variable as `export JULIA_PROJECT="$PWD"`)
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

## Multi-threadding
So far, we have been able to decrese the computation from 39ms to something like 13ms. Can we improve? Let's now turn our attention to multi-threadding, where we will not pay the penalty for IPC. Moreover, the computation of Julia set is multi-thread friendly, as all the memory can be pre-allocatted. We slightly modify our code to accept different methods distributing the work among slices in the pre-allocated matrix. To start Julia with support of multi-threadding, run it with `julia -t n`, where `n` is the number of threads. It is reccomended to set `n` to number of physical cores, since in hyper-threadding two threads shares arithmetic units of a single core, and in applications for which Julia was built, they are usually saturated.
```julia
using BenchmarkTools
function juliaset_pixel(z₀, c)
    z = z₀
    for i in 1:255
        abs2(z)> 4.0 && return (i - 1)%UInt8
        z = z*z + c
    end
    return UInt8(255)
end

function juliaset_column!(img, c, n, j)
    x = -2.0 + (j-1)*4.0/(n-1)
    for i in 1:n
        y = -2.0 + (i-1)*4.0/(n-1)
        @inbounds img[i,j] = juliaset_pixel(x+im*y, c)
    end
    nothing
end

function juliaset(x, y, n=1000)
    c = x + y*im
    img = Array{UInt8,2}(undef,n,n)
    for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    return img
end

julia> @btime juliaset(-0.79, 0.15, 1000);
   38.932 ms (2 allocations: 976.67 KiB)
```
Let's now try to speed-up the calculation using multi-threadding. `Julia v0.5` has introduced multi-threadding with static-scheduller with a simple syntax: just prepend the for-loop with a `Threads.@threads` macro. With that, the first multi-threaded version will looks like
```julia
function juliaset_static(x, y, n=1000)
    c = x + y*im
    img = Array{UInt8,2}(undef,n,n)
    Threads.@threads for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    return img
end
```
with benchmark
```
julia> 	@btime juliaset_static(-0.79, 0.15, 1000);
  15.751 ms (27 allocations: 978.75 KiB)
```
Although we have used four-threads, and the communication overhead should be next to zero, the speed improvement is ``2.4``. Why is that? 

To understand bettern what is going on, we have improved the profiler we have been developing last week. The logging profiler logs time of entering and exitting every function call of every thread, which is useful to understand, what is going on. The api is not yet polished, but it will do its job. Importantly, to prevent excessive logging, we ask to log only some functions.
```julia
using LoggingProfiler
function juliaset_static(x, y, n=1000)
    c = x + y*im
    img = Array{UInt8,2}(undef,n,n)
    Threads.@threads :dynamic for j in 1:n
        LoggingProfiler.@recordfun juliaset_column!(img, c, n, j)
    end
    return img
end

LoggingProfiler.initbuffer!(1000)
juliaset_static(-0.79, 0.15, 1000);
LoggingProfiler.recorded()
LoggingProfiler.adjustbuffer!()
juliaset_static(-0.79, 0.15, 1000)
LoggingProfiler.export2svg("/tmp/profile.svg")
LoggingProfiler.export2luxor("profile.png")
```
![profile.png](profile.png)
From the visualization of the profiler we can see not all threads were working the same time. Thread 1 and 4 were working less that Thread 2 and 3. The reason is that the static scheduller partition the total number of columns (1000) into equal parts, where the total number of parts is equal to the number of threads, and assign each to a single thread. In our case, we will have four parts each of size 250. Since execution time of computing value of each pixel is not the same, threads with a lot zero iterations will finish considerably faster. This is the incarnation of one of the biggest problems in multi-threadding / schedulling. A contemprary approach is to switch to  dynamic schedulling, which divides the problem into smaller parts, and when a thread is finished with one part, it assigned new not-yet computed part.

From 1.5, one can specify the scheduller for `Threads.@thread [scheduller] for` construct to be either `:static` and / or `:dynamic`. The `:dynamic` is compatible with the `partr` dynamic scheduller. From `1.8`, `:dynamic` is default, but the range is dividided into `nthreads()` parts, which is the reason why we do not see an improvement.

Dynamic scheduller is also supported using by `Threads.@spawn` macro. The prototypical approach used for invocation is the fork-join model, where one recursivelly partitions the problems and wait in each thread for the other
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
function juliaset_forkjoin(x, y, n=1000, ntasks = 16)
    c = x + y*im
    img = Array{UInt8,2}(undef,n,n)
    juliaset_recspawn!(img, c, n, 1, n, ntasks)
    return img
end

julia> @btime juliaset_forkjoin(-0.79, 0.15);
  10.326 ms (142 allocations: 986.83 KiB)
```
This is so far our fastest construction with speedup `38.932 / 10.326 = 3.77×`.

Unfortunatelly, the `LoggingProfiler` does not handle task migration at the moment, which means that we cannot visualize the results. Due to task switching overhead, increasing the granularity might not pay off.
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
function juliaset_folds(x, y, n=1000, basesize = 2)
    c = x + y*im
    img = Array{UInt8,2}(undef,n,n)
    @floop ThreadedEx(basesize = basesize) for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    return img
end

julia> @btime juliaset_folds(-0.79, 0.15, 1000);
  10.253 ms (3960 allocations: 1.24 MiB)
```
where `basesize` is the size of the smallest part allocated to a single thread, in this case 2 columns.
```julia
julia> @btime juliaset_folds(-0.79, 0.15, 1000);
  10.575 ms (52 allocations: 980.12 KiB)
```

```julia
function juliaset_folds(x, y, n=1000, basesize = 2)
    c = x + y*im
    img = Array{UInt8,2}(undef,n,n)
    @floop DepthFirstEx(basesize = basesize) for j in 1:n
        juliaset_column!(img, c, n, j)
    end
    return img
end
julia> @btime juliaset_folds(-0.79, 0.15, 1000);
  10.421 ms (3582 allocations: 1.20 MiB)
```

We can identify the best smallest size of the work `basesize` and measure its influence on the time
```julia
map(2 .^ (0:7)) do bs 
	t = @belapsed juliaset_folds(-0.79, 0.15, 1000, $(bs));
	(;basesize = bs, time = t)
end |> DataFrame
```

```julia
 Row │ basesize  time
     │ Int64     Float64
─────┼─────────────────────
   1 │        1  0.0106803
   2 │        2  0.010267
   3 │        4  0.0103081
   4 │        8  0.0101652
   5 │       16  0.0100204
   6 │       32  0.0100097
   7 │       64  0.0103293
   8 │      128  0.0105411
```
We observe that the minimum is for `basesize = 32`, for which we got `3.8932×` speedup. 

## Garbage collector is single-threadded
Keep reminded that while threads are very easy very convenient to use, there are use-cases where you might be better off with proccess, even though there will be some communication overhead. One such case happens when you need to allocate and free a lot of memory. This is because Julia's garbage collector is single-threadded. Imagine a task of making histogram of bytes in a directory.
For a fair comparison, we will use `Transducers`, since they offer thread and process based paralelism
```julia
using Transducers
@everywhere begin 
	function histfile(filename)
		h = Dict{UInt8,Int}()
		foreach(open(read, filename, "r")) do b 
			h[b] = get(h, b, 0) + 1
		end
		h
	end
end

files = filter(isfile, readdir("/Users/tomas.pevny/Downloads/", join = true))
@elapsed foldxd(mergewith(+), files |> Map(histfile))
150.863183701
```
and using the multi-threaded version of `map`
```julia
@elapsed foldxt(mergewith(+), files |> Map(histfile))
205.309952618
```
we see that the threadding is actually worse than process based paralelism despite us paying the price for serialization  and deserialization of  `Dict`. Needless to say that changing `Dict` to `Vector` as
```julia
using Transducers
@everywhere begin 
	function histfile(filename)
		h = Dict{UInt8,Int}()
		foreach(open(read, filename, "r")) do b 
			h[b] = get(h, b, 0) + 1
		end
		h
	end
end
files = filter(isfile, readdir("/Users/tomas.pevny/Downloads/", join = true))
@elapsed foldxd(mergewith(+), files |> Map(histfile))
86.44577969
@elapsed foldxt(mergewith(+), files |> Map(histfile))
105.32969331
```
is much better.


## Locks / lock-free multi-threadding
Avoid locks.

## Take away message
When deciding, what kind of paralelism to employ, consider following
- for tightly coupled computation over shared data, multi-threadding is more suitable due to non-existing sharing of data between processes
- but if the computation requires frequent allocation and freeing of memery, or IO, separate processes are multi-suitable, since garbage collectors are independent between processes
- Making all cores busy while achieving an ideally linear speedup is difficult and needs a lot of experience and knowledge. Tooling and profilers supporting debugging of parallel processes is not much developped.
- `Transducers` thrives for (almost) the same code to support thread- and process-based paralelism.

### Materials
- [http://cecileane.github.io/computingtools/pages/notes1209.html](http://cecileane.github.io/computingtools/pages/notes1209.html)
- [https://lucris.lub.lu.se/ws/portalfiles/portal/61129522/julia_parallel.pdf](https://lucris.lub.lu.se/ws/portalfiles/portal/61129522/julia_parallel.pdf)
- [http://igoro.com/archive/gallery-of-processor-cache-effects/](http://igoro.com/archive/gallery-of-processor-cache-effects/)
- [https://www.csd.uwo.ca/~mmorenom/cs2101a_moreno/Parallel_computing_with_Julia.pdf](https://www.csd.uwo.ca/~mmorenom/cs2101a_moreno/Parallel_computing_with_Julia.pdf)
- Complexity of thread schedulling [https://www.youtube.com/watch?v=YdiZa0Y3F3c](https://www.youtube.com/watch?v=YdiZa0Y3F3c)
- TapIR --- Teaching paralelism to Julia compiler [https://www.youtube.com/watch?v=-JyK5Xpk7jE](https://www.youtube.com/watch?v=-JyK5Xpk7jE)
- Threads: [https://juliahighperformance.com/code/Chapter09.html](https://juliahighperformance.com/code/Chapter09.html)
- Processes: [https://juliahighperformance.com/code/Chapter10.html](https://juliahighperformance.com/code/Chapter10.html)
- Alan Adelman uses FLoops in [https://www.youtube.com/watch?v=dczkYlOM2sg](https://www.youtube.com/watch?v=dczkYlOM2sg)
- Examples: ?Heat equation? from [https://hpc.llnl.gov/training/tutorials/](introduction-parallel-computing-tutorial#Examples(https://hpc.llnl.gov/training/tutorials/)