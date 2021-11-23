# Generated functions
using Dictionaries
function retrieve_code_info(sigtypes, world = Base.get_world_counter())
    S = Tuple{map(s -> Core.Compiler.has_free_typevars(s) ? typeof(s.parameters[1]) : s, sigtypes)...}
    _methods = Base._methods_by_ftype(S, -1, world)
    if isempty(_methods) 
        @info("method $(sigtypes) does not exist, may-be run it once")
        return(nothing)
    end
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
    offset = 0
    for i in 1:calls.i[]
        offset -= calls.startstop[i] == :stop
        foreach(_ -> print(io, " "), 1:max(offset, 0))
        rel_time = calls.stamps[i] - calls.stamps[1]
        println(io, calls.event[i], ": ", rel_time)
        offset += calls.startstop[i] == :start
    end
end

global const to = Calls(100)

function record_start(ev::Symbol)
    calls = Main.to
    n = calls.i[] = calls.i[] + 1
    n > length(calls.stamps) && return 
    calls.event[n] = ev
    calls.startstop[n] = :start
    calls.stamps[n] = time_ns()
end

function record_end(ev::Symbol)
    t = time_ns()
    calls = Main.to
    n = calls.i[] = calls.i[] + 1
    n > length(calls.stamps) && return 
    calls.event[n] = ev
    calls.startstop[n] = :stop
    calls.stamps[n] = t
end

reset!(calls::Calls) = calls.i[] = 0


function overdubbable(ex::Expr) 
    ex.head != :call && return(false)
    length(ex.args) < 2 && return(false)
    (ex.args[1] isa Core.IntrinsicFunction) && return(false)
    return(true)
end
# overdubbable(ex::Expr) = false
overdubbable(ex) = false
# overdubbable(ctx::Context, ex::Expr) = ex.head == :call
# overdubbable(ctx::Context, ex) = false
# timable(ctx::Context{Nothing}, ex) = true
timable(ex::Expr) = ex.head == :call
timable(ex) = false

rename_args(ex, slotvar, ssamap) = ex
rename_args(c::Core.GotoIfNot, slotvar, ssamap) = Core.GotoIfNot(rename_args(c.cond, slotvar, ssamap), ssamap[c.dest])
rename_args(ex::Expr, slotvar, ssamap) = Expr(ex.head, rename_args(ex.args, slotvar, ssamap)...)
rename_args(args::AbstractArray, slotvar, ssamap) = map(a -> rename_args(a, slotvar, ssamap), args)
rename_args(r::Core.ReturnNode, slotvar, ssamap) = Core.ReturnNode(rename_args(r.val, slotvar, ssamap))
rename_args(a::Core.SlotNumber, slotvar, ssamap) = slotvar[a.id]
rename_args(a::Core.SSAValue, slotvar, ssamap) = Core.SSAValue(ssamap[a.id])

exportname(ex::GlobalRef) = QuoteNode(ex.name)
exportname(ex::Expr) = exportname(ex.args[1])
exportname(i::Int) = QuoteNode(Symbol("Int(",i,")"))

using Base: invokelatest
dummy() = return

overdub(f::Core.IntrinsicFunction, args...) = f(args...)

@generated function overdub(f::F, args...) where {F}
    @show (F, args...)
    ci = retrieve_code_info((F, args...))
    if ci === nothing 
        @show Expr(:call, :f, [:(args[$(i)]) for i in 1:length(args)]...)
        return(Expr(:call, :f, [:(args[$(i)]) for i in 1:length(args)]...))
    end
    # this is to initialize a new CodeInfo and fill it with values from the 
    # overdubbed function
    new_ci = code_lowered(dummy, Tuple{})[1]
    empty!(new_ci.code)
    empty!(new_ci.slotnames)
    foreach(s -> push!(new_ci.slotnames, s), ci.slotnames)
    new_ci.slotnames = vcat([Symbol("#self#"), :f, :args], ci.slotnames[length(args)+2:end])
    new_ci.slotflags = vcat([0x00, 0x00, 0x00], ci.slotflags[length(args)+2:end])
    empty!(new_ci.linetable)
    foreach(s -> push!(new_ci.linetable, s), ci.linetable)
    empty!(new_ci.codelocs)

    ssamap = Dict{Int, Int}()
    slotvar = Dict{Int, Any}()
    for i in 1:length(args)
        push!(new_ci.code, Expr(:call, Base.getindex, Core.SlotNumber(3), i))
        slotvar[i+1] = Core.SSAValue(i)
        push!(new_ci.codelocs, ci.codelocs[1])
    end
    slotvar[1] = Core.SlotNumber(1)
    foreach(i -> slotvar[i[2]] = Core.SlotNumber(i[1]+3), enumerate(length(args)+2:length(ci.slotnames)))

    j = length(args)
    for (i, ex) in enumerate(ci.code)
        if timable(ex)
            fname = exportname(ex)
            push!(new_ci.code, Expr(:call, GlobalRef(Main, :record_start), fname))
            push!(new_ci.codelocs, ci.codelocs[i])
            j += 1
            # ex = overdubbable(ex) ? Expr(:call, :overdub, ex.args...) : ex
            # ex = overdubbable(ex) ? Expr(:call, GlobalRef(Main, :overdub), ex.args...) : ex
            push!(new_ci.code, ex)
            push!(new_ci.codelocs, ci.codelocs[i])
            j += 1
            ssamap[i] = j
            push!(new_ci.code, Expr(:call, GlobalRef(Main, :record_end), fname))
            push!(new_ci.codelocs, ci.codelocs[i])
            j += 1
        else
            push!(new_ci.code, ex)
            push!(new_ci.codelocs, ci.codelocs[i])
            j += 1
            ssamap[i] = j
        end
    end
    for i in length(args)+1:length(new_ci.code)
        new_ci.code[i] = rename_args(new_ci.code[i], slotvar, ssamap)
    end
    new_ci

    # Core.Compiler.replace_code_newstyle!(ci, ir, length(ir.argtypes)-1)
    new_ci.inferred = false
    new_ci.ssavaluetypes = length(new_ci.code)
    # new_ci
    # new_ci
    return(new_ci)
end

function foo(x, y)
   z = x * y
   z + sin(y)
end

reset!(to)
overdub(foo, 1.0, 1.0)
overdub(sin, 1.0)
to

function overdub2(::typeof(foo), args...)
    x = args[1]
    y = args[2]
    push!(to, :start, :fun)
    z = x * y
    push!(to, :stop, :fun)
    push!(to, :start, :fun)
    r = z + sin(y)
    push!(to, :stop, :fun)
    r
end

ci = retrieve_code_info((typeof(overdub2), typeof(foo), Float64, Float64))

