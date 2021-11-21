# Generated functions
using Dictionaries
function retrieve_code_info(sigtypes, world = Base.get_world_counter())
  S = Tuple{map(s -> Core.Compiler.has_free_typevars(s) ? typeof(s.parameters[1]) : s, sigtypes)...}
  _methods = Base._methods_by_ftype(S, -1, world)
  isempty(_methods) && @error("method $(sigtypes) does not exist, may-be run it once")
  type_signature, raw_static_params, method = _methods[1] # method is the same as we would get by invoking methods(+, (Int, Int)).ms[1]  

  # this provides us with the CodeInfo
  method_instance = Core.Compiler.specialize_method(method, type_signature, raw_static_params, false)
  code_info = Core.Compiler.retrieve_code_info(method_instance)
end

struct Calls
    stamps::Vector{Float64} # contains the time stamps
    event::Vector{Symbol}  # name of the function that is being recorded
    startstop::Vector{Symbol} # if the time stamp corresponds to start or to stop
    i::Ref{Int}
end

function Calls(n::Int)
    Calls(Vector{Float64}(undef, n+1), Vector{Symbol}(undef, n+1), Vector{Symbol}(undef, n+1), Ref{Int}(0))
end

function Base.show(io::IO, calls::Calls)
    for i in 1:calls.i[]
        println(io, calls.stamps[i] - calls.stamps[1],"  ", calls.startstop[i],"  ",calls.event[i])
    end
end

function Base.push!(calls::Calls, s::Symbol, ev::Symbol)
    n = calls.i[] = calls.i[] + 1
    n > length(calls.stamps) && return 
    calls.event[n] = ev
    calls.startstop[n] = s
    calls.stamps[n] = time()
end

reset!(calls::Calls) = calls.i[] = 0

struct Context{T<:Union{Nothing, Vector{Symbol}}}
    functions::T
end
Context() = Context(nothing)

ctx = Context()

function overdubbable(ex::Expr) 
    ex.head != :call && return(false)
    length(ex.args) < 2 && return(false)
    (ex.args[1] isa Core.IntrinsicFunction) && return(false)
    return(true)
end
overdubbable(ex::Expr) = false
overdubbable(ex) = false
# overdubbable(ctx::Context, ex::Expr) = ex.head == :call
# overdubbable(ctx::Context, ex) = false
# timable(ctx::Context{Nothing}, ex) = true
timable(ex::Expr) = ex.head == :call
timable(ex) = false

function foo(x, y)
   z = x * y
   z + sin(y)
end

rename_args(ex, slot_vars, ssa_vars) = ex
rename_args(ex::Expr, slot_vars, ssa_vars) = Expr(ex.head, rename_args(ex.args, slot_vars, ssa_vars)...)
rename_args(args::AbstractArray, slot_vars, ssa_vars) = map(a -> rename_args(a, slot_vars, ssa_vars), args)
rename_args(r::Core.ReturnNode, slot_vars, ssa_vars) = Core.ReturnNode(rename_args(r.val, slot_vars, ssa_vars))
rename_args(a::Core.SlotNumber, slot_vars, ssa_vars) = slot_vars[a.id]
rename_args(a::Core.SSAValue, slot_vars, ssa_vars) = ssa_vars[a.id]

assigned_vars(ex) = []
assigned_vars(ex::Expr) = assigned_vars(ex.args)
assigned_vars(args::AbstractArray) = mapreduce(assigned_vars, vcat, args)
assigned_vars(r::Core.ReturnNode) = assigned_vars(r.val)
assigned_vars(a::Core.SlotNumber) = []
assigned_vars(a::Core.SSAValue) = [a.id]

exportname(ex::GlobalRef) = ex.name
exportname(ex::Expr) = ex.args[1]

overdub(ctx::Context, f::Core.IntrinsicFunction, args...) = f(args...)

@generated function overdub(ctx::Context, f::F, args...) where {F}
    ci = retrieve_code_info((F, args...))
    slot_vars = Dict(enumerate(ci.slotnames))
    # ssa_vars = Dict(i => gensym(:left) for i in 1:length(ci.code))
    ssa_vars = Dict(i => Symbol(:L, i) for i in 1:length(ci.code))
    used = assigned_vars(ci.code) |> distinct
    exprs = []
    for i in 1:length(args)
        push!(exprs, Expr(:(=), ci.slotnames[i+1], :(args[$(i)])))
    end
    for (i, ex) in enumerate(ci.code)
        ex = rename_args(ex, slot_vars, ssa_vars)
        @show ex
        if ex isa Core.ReturnNode 
            push!(exprs, Expr(:return, ex.val))
            continue
        end
        if timable(ex)
            fname = exportname(ex)
            fname = :(Symbol($(fname)))
            push!(exprs, Expr(:call, :push!, :to, :(:start), fname))
            ex = overdubbable(ex) ? Expr(:call, :overdub, :ctx, ex.args...) : ex
            ex = i ∈ used ? Expr(:(=) , ssa_vars[i], ex) : ex
            push!(exprs, ex)
            push!(exprs, Expr(:call, :push!, :to, :(:stop), fname))
        else
            ex = i ∈ used ? Expr(:(=) , ssa_vars[i], ex) : ex
            push!(exprs, ex)
        end
    end
    r = Expr(:block, exprs...)
    @show r 
    # println("  ")
    r
end


global const to = Calls(100)
reset!(to)
ctx = Context()
overdub(ctx, foo, 1.0, 1.0)


