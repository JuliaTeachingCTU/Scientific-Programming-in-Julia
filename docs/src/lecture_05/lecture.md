# [Benchmarking, profiling, and performance gotchas](@id perf_lecture)
- write the code well from the start -> unit test -> start thinking about performance
	+ danger of premature optimization
- human intuition is bad when reasoning about where the program spends most of the time
	+ how to identify them <- profiler (intro to sampling based profilers)
- performance tweaks
	+ effect of global variables
	+ memory allocations matters (mainly on heap - theoretical intro to this topic)
	+ differnce between named tuple and dict
	+ boxing in closures
	+ IO
	+ memory layout matters

- create a list of examples for the lecture