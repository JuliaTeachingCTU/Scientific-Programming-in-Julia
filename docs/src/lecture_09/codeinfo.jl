# Generated functions
using Dictionaries
include("calls.jl")

function retrieve_code_info(sigtypes, world = Base.get_world_counter())
    S = Tuple{map(s -> Core.Compiler.has_free_typevars(s) ? typeof(s.parameters[1]) : s, sigtypes)...}
    _methods = Base._methods_by_ftype(S, -1, world)
    if isempty(_methods) 
        @info("method $(sigtypes) does not exist")
        return(nothing)
    end
    type_signature, raw_static_params, method = _methods[1]
    method_instance = Core.Compiler.specialize_method(method, type_signature, raw_static_params, false)
    code_info = Core.Compiler.retrieve_code_info(method_instance)
end

function overdubbable(ex::Expr) 
    ex.head != :call && return(false)
    length(ex.args) < 2 && return(false)
    ex.args[1] isa Core.GlobalRef && return(true)
    ex.args[1] isa Symbol && return(true)
    return(false)
end

# overdubbable(ex::Expr) = false
overdubbable(ex) = false
# overdubbable(ctx::Context, ex::Expr) = ex.head == :call
# overdubbable(ctx::Context, ex) = false
# timable(ctx::Context{Nothing}, ex) = true
timable(ex::Expr) = overdubbable(ex)
timable(ex) = false

#
remap(ex::Expr, maps) = Expr(ex.head, remap(ex.args, maps)...)
remap(args::AbstractArray, maps) = map(a -> remap(a, maps), args)
remap(c::Core.GotoNode, maps) = Core.GotoNode(maps.goto[c.label])
remap(c::Core.GotoIfNot, maps) = Core.GotoIfNot(remap(c.cond, maps), maps.goto[c.dest])
remap(r::Core.ReturnNode, maps) = Core.ReturnNode(remap(r.val, maps))
remap(a::Core.SlotNumber, maps) = maps.slots[a.id]
remap(a::Core.SSAValue, maps) = Core.SSAValue(maps.ssa[a.id])
remap(a::Core.NewvarNode, maps) = Core.NewvarNode(maps.slots[a.slot.id])
remap(a::GlobalRef, maps) = a
remap(a::QuoteNode, maps) = a
remap(ex, maps) = ex

# remove static parameters (see https://discourse.julialang.org/t/does-overdubbing-in-generated-function-inserts-inlined-code/71868)
remove_static(ex)  = ex
function remove_static(ex::Expr)
    ex.head != :call && return(ex)
    length(ex.args) != 2 && return(ex)
    !(ex.args[1] isa Expr) && return(ex)
    (ex.args[1].head == :static_parameter) && return(ex.args[2])
    return(ex)
end

exportname(ex::GlobalRef) = QuoteNode(ex.name)
exportname(ex::Symbol) = QuoteNode(ex)
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
    empty!(new_ci.linetable)
    empty!(new_ci.codelocs)

    new_ci.slotnames = vcat([Symbol("#self#"), :f, :args], ci.slotnames[2:end])
    new_ci.slotflags = vcat([0x00, 0x00, 0x00], ci.slotflags[2:end])
    foreach(s -> push!(new_ci.linetable, s), ci.linetable)

    maps = (
        ssa = Dict{Int, Int}(),
        slots = Dict{Int, Any}(),
        goto = Dict{Int,Int}(),
        )

    #we need to map indexes of slot-variables from ci to their new values. 
    # except the first one, we just remap them
    maps.slots[1] = Core.SlotNumber(1)
    foreach(i -> maps.slots[i] = Core.SlotNumber(i + 2), 2:length(ci.slotnames)) # they are shifted by 2 accomondating inserted `f` and `args`
    @assert all(ci.slotnames[i] == new_ci.slotnames[maps.slots[i].id] for i in 1:length(ci.slotnames))  #test that the remapping is right

    #if somewhere the original parameters of the functions will be used 
    #they needs to be remapped to an SSAValue from here, since the overdubbed
    # function has signatures overdub(f, args...) instead of f(x,y,z...)
    ssa_no = 0
    for i in 1:length(args)
        ssa_no +=1
        push!(new_ci.code, Expr(:call, Base.getindex, Core.SlotNumber(3), i))
        maps.slots[i+1] = Core.SSAValue(ssa_no)
        push!(new_ci.codelocs, ci.codelocs[1])
    end

    for (ci_line, ex) in enumerate(ci.code)
        if timable(ex)
            fname = exportname(ex)
            push!(new_ci.code, Expr(:call, GlobalRef(Main, :record_start), fname))
            push!(new_ci.codelocs, ci.codelocs[ci_line])
            ssa_no += 1
            maps.goto[ci_line] = ssa_no
            ex = overdubbable(ex) ? Expr(:call, GlobalRef(Main, :overdub), ex.args...) : ex
            push!(new_ci.code, ex)
            push!(new_ci.codelocs, ci.codelocs[ci_line])
            ssa_no += 1
            maps.ssa[ci_line] = ssa_no
            push!(new_ci.code, Expr(:call, GlobalRef(Main, :record_end), fname))
            push!(new_ci.codelocs, ci.codelocs[ci_line])
            ssa_no += 1
        else
            push!(new_ci.code, ex)
            push!(new_ci.codelocs, ci.codelocs[ci_line])
            ssa_no += 1
            maps.ssa[ci_line] = ssa_no
        end
    end

    for i in length(args)+1:length(new_ci.code)
        ex = remove_static(new_ci.code[i])
        new_ci.code[i] = remap(ex, maps)
    end
    new_ci

    # Core.Compiler.replace_code_newstyle!(ci, ir, length(ir.argtypes)-1)
    new_ci.inferred = false
    new_ci.ssavaluetypes = length(new_ci.code)
    # new_ci
    return(new_ci)
end

function foo(x, y)
   z =  x * y
   z + sin(y)
end


reset!(to)
overdub(foo, 1.0, 1.0)
new_ci = overdub(sin, 1.0)
to

# Seems like now, I am crashing here
# typeof(Base._promote), Irrational{:π}, Int64)


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


function test(x::T) where T<:Union{Float64, Float32}
    x < T(pi)
end
ci = retrieve_code_info((typeof(test), Float64))


function overdub_test(::typeof(test), args...)
    x = args[1]
    T = eltype(x)
    x < T(pi)
end

ci = retrieve_code_info((typeof(overdub_test), typeof(test), Float64))

