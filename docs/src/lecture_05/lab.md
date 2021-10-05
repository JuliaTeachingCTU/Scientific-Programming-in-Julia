# Lab 05: Benchmarking, profiling and performance gotchas



```@repl lab05_polynomial
function polynomial(a, x)
    accumulator = 0
    for i in length(a):-1:1
        accumulator += x^(i-1) * a[i] # ! 1-based indexing for arrays
    end
    return accumulator
end

a = [-19, 7, -4, 6]
af = Float32.(a)
x = 3
```

```@repl lab05_polynomial
using InteractiveUtils
@code_warntype polynomial(a, x)  # type stable
@code_warntype polynomial(af, x) # type unstable
```

```@repl lab05_polynomial
using Profile, ProfileSVG
using BenchmarkTools

a = rand(-20:20, 1000); # slightly longer polynomial
# for i in 1:length(a)
#     @btime polynomial($(a[1:i]), $x)
# end

@profview polynomial(a, x)
```

```@repl lab05_polynomial
function polynomial(a, x::AbstractMatrix)
    accumulator = zeros(eltype(x), size(x))
    for i in length(a):-1:1
        accumulator += x^(i-1) * a[i]
    end
    return accumulator
end

a = rand(-20:20, 100);
A = rand(100,100)

@profview polynomial(a, A)
ProfileSVG.save("./prof.svg")
```



```@repl lab05_polynomial
run_polynom = () -> begin for _ in 1:10000 
```



