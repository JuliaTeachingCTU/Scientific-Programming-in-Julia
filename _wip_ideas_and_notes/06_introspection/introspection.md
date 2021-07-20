https://www.youtube.com/watch?v=mSgXWpvQEHE

### Language introspection
  - Different levels of compilation
  - What AST
  - Manipulating AST
  - **LABS:**
    + Removing boundary checks
    + generate call graph of a code (TreeView.jl)


## Removing boilerplate from a code

An interesting usecase for metaprogramming is removing "builerplate" code. Imagine we are exposing a rest api. There is a list of URLs and a list of functions. You have a complicated function handling all the error checking etc. 

```julia
query_f(v::Vector, service, parsing_fun; timeout=5) = map(d -> query_f(JSON.json(d), service, parsing_fun; timeout), v)
query_f(d::Dict, service, parsing_fun; timeout=5) = query_f(JSON.json(d), service, parsing_fun; timeout)
function query_f(q::String, service, parsing_fun; timeout)
    r = HTTP.post(service(timeout), ["Content-Type" => "application/json"], q, status_exception=false)
    if r.status != 200
        @warn "service timeouted/returned an error"
        @show r
    else
        parsing_fun(String(r.body))
    end
end
```

for fun_name in ["a","b","c","d"]
    eval(:($(fun_name)(v;timeout = 5) = query_f(v, x -> "http://example.com:8009/v1.0/api/$(fun_name)/${x}", JSON.parse ;timeout)))
end


# We can do a similar example for something like blas_exposition / definition of transposition
