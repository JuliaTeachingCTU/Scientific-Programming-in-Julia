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
    mi = Core.Compiler.specialize_method(method, type_signature, raw_static_params, false)
    ci = Base.isgenerated(mi) ? Core.Compiler.get_staged(mi) : Base.uncompressed_ast(method)
    Base.Meta.partially_inline!(ci.code, [], method.sig, Any[raw_static_params...], 0, 0, :propagate)
    ci
end

# # We migth consider this from IRTools. Importantly, it has partially_inline to get 
# # rid of static parameters
# function meta(T; types = T, world = worldcounter())
#   F = T.parameters[1]
#   F == typeof(invoke) && return invoke_meta(T; world = world)
#   F isa DataType && (F.name.module === Core.Compiler ||
#                      F <: Core.Builtin ||
#                      F <: Core.Builtin) && return nothing
#   _methods = Base._methods_by_ftype(T, -1, world)
#   length(_methods) == 0 && return nothing
#   type_signature, sps, method = last(_methods)
#   sps = svec(map(untvar, sps)...)
#   @static if VERSION >= v"1.2-"
#     mi = Core.Compiler.specialize_method(method, types, sps)
#     ci = Base.isgenerated(mi) ? Core.Compiler.get_staged(mi) : Base.uncompressed_ast(method)
#   else
#     mi = Core.Compiler.code_for_method(method, types, sps, world, false)
#     ci = Base.isgenerated(mi) ? Core.Compiler.get_staged(mi) : Base.uncompressed_ast(mi)
#   end
#   Base.Meta.partially_inline!(ci.code, [], method.sig, Any[sps...], 0, 0, :propagate)
#   Meta(method, mi, ci, method.nargs, sps)
# end


function overdubbable(ex::Expr) 
    ex.head != :call && return(false)
    length(ex.args) < 2 && return(false)
    ex.args[1] isa Core.GlobalRef && return(true)
    ex.args[1] isa Symbol && return(true)
    return(false)
end

overdubbable(ex) = false
timable(ex) = overdubbable(ex)

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

exportname(ex::GlobalRef) = QuoteNode(ex.name)
exportname(ex::Symbol) = QuoteNode(ex)
exportname(ex::Expr) = exportname(ex.args[1])
exportname(i::Int) = QuoteNode(Symbol("Int(",i,")"))

dummy() = return
function empty_codeinfo()
    new_ci = code_lowered(dummy, Tuple{})[1]
    empty!(new_ci.code)
    empty!(new_ci.slotnames)
    empty!(new_ci.linetable)
    empty!(new_ci.codelocs)
    new_ci
end


overdub(f::Core.IntrinsicFunction, args...) = f(args...)

@generated function overdub(f::F, args...) where {F}
    @show (F, args...)
    ci = retrieve_code_info((F, args...))
    if ci === nothing 
        return(Expr(:call, :f, [:(args[$(i)]) for i in 1:length(args)]...))
    end

    new_ci = empty_codeinfo()
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
    newci_no = 0
    for i in 1:length(args)
        newci_no +=1
        push!(new_ci.code, Expr(:call, Base.getindex, Core.SlotNumber(3), i))
        maps.slots[i+1] = Core.SSAValue(newci_no)
        push!(new_ci.codelocs, ci.codelocs[1])
    end

    for (ci_no, ex) in enumerate(ci.code)
        if timable(ex)
            fname = exportname(ex)
            push!(new_ci.code, Expr(:call, GlobalRef(Main, :record_start), fname))
            push!(new_ci.codelocs, ci.codelocs[ci_no])
            newci_no += 1
            maps.goto[ci_no] = newci_no
            # @show ex
            # @show Expr(:call, GlobalRef(Main, :overdub), ex.args...)
            # ex = overdubbable(ex) ? Expr(:call, GlobalRef(Main, :overdub), ex.args...) : ex
            # ex = overdubbable(ex) ? Expr(:call, GlobalRef(Main, :overdub), ex.args...) : ex
            push!(new_ci.code, ex)
            push!(new_ci.codelocs, ci.codelocs[ci_no])
            newci_no += 1
            maps.ssa[ci_no] = newci_no
            push!(new_ci.code, Expr(:call, GlobalRef(Main, :record_end), fname))
            push!(new_ci.codelocs, ci.codelocs[ci_no])
            newci_no += 1
        else
            push!(new_ci.code, ex)
            push!(new_ci.codelocs, ci.codelocs[ci_no])
            newci_no += 1
            maps.ssa[ci_no] = newci_no
        end
    end

    for i in length(args)+1:length(new_ci.code)
        new_ci.code[i] = remap(new_ci.code[i], maps)
    end
    new_ci
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
to
reset!(to)
new_ci = overdub(sin, 1.0)
to
