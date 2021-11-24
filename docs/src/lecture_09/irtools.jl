# Generated functions
using Dictionaries, IRTools
include("calls.jl")
using IRTools: var, xcall, insert!, insertafter!, func

function timable(ex::Expr) 
    ex.head != :call && return(false)
    length(ex.args) < 2 && return(false)
    ex.args[1] isa Core.GlobalRef && return(true)
    ex.args[1] isa Symbol && return(true)
    return(false)
end
timable(ex) = false

exportname(ex::GlobalRef) = QuoteNode(ex.name)
exportname(ex::Symbol) = QuoteNode(ex)
exportname(ex::Expr) = exportname(ex.args[1])
exportname(i::Int) = QuoteNode(Symbol("Int(",i,")"))

function foo(x, y)
   z =  x * y
   z + sin(y)
end

# ir = @code_ir foo(1.0, 1.0)
ir = @code_ir sin(1.0)

# writing our profiler would be relatively 
# we will iterate over the ir code and inserts appropriate logs
for b in IRTools.blocks(ir)
    for (v, ex) in b
        if timable(ex.expr)
            fname = exportname(ex.expr)
            insert!(b, v, xcall(Main, :record_start, fname))
            insertafter!(b, v, xcall(Main, :record_end, fname))
        end
    end
end

@generated function profile_fun(f, args...)
    m = IRTools.Inner.meta(Tuple{f,args...})
    ir = IRTools.Inner.IR(m)
    for b in IRTools.blocks(ir)
        for (v, ex) in b
            if timable(ex.expr)
                fname = exportname(ex.expr)
                insert!(b, v, xcall(Main, :record_start, fname))
                insertafter!(b, v, xcall(Main, :record_end, fname))
            end
        end
    end

    # we need to deal with the problem that ir has different set so f arguments than profile_fun(f, args...)
    # THis is what a dynamo does for us
    return(IRTools.Inner.build_codeinfo(ir))
end

f = func(ir)
# f(nothing, 1.0, 1.0)
# f = func(ir)
f(nothing, 1.0)

function func(m::Module, ir::IR)
  @eval @generated function $(gensym())($([Symbol(:arg, i) for i = 1:length(arguments(ir))]...))
    return build_codeinfo($ir)
  end
end
