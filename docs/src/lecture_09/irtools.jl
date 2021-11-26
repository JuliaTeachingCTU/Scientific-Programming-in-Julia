# Generated functions
using IRTools
using IRTools: var, xcall, insert!, insertafter!, func, recurse!, @dynamo
include("loggingprofiler.jl")
LoggingProfiler.resize!(LoggingProfiler.to, 10000)

function timable(ex::Expr) 
    ex.head != :call && return(false)
    length(ex.args) < 2 && return(false)
    ex.args[1] isa Core.GlobalRef && return(true)
    ex.args[1] isa Symbol && return(true)
    return(false)
end
timable(ex) = false

recursable(gr::GlobalRef) = gr.name ∉ [:profile_fun, :record_start, :record_end]
recursable(ex::Expr) = ex.head == :call && recursable(ex.args[1])
recursable(ex) = false

exportname(ex::GlobalRef) = QuoteNode(ex.name)
exportname(ex::Symbol) = QuoteNode(ex)
exportname(ex::Expr) = exportname(ex.args[1])
exportname(i::Int) = QuoteNode(Symbol("Int(",i,")"))

profile_fun(f::Core.IntrinsicFunction, args...) = f(args...)
profile_fun(f::Core.Builtin, args...) = f(args...)

@dynamo function profile_fun(f, args...)
    ir = IRTools.Inner.IR(f, args...)
    for (v, ex) in ir
        if timable(ex.expr)
            fname = exportname(ex.expr)
            insert!(ir, v, xcall(LoggingProfiler, :record_start, fname))
            insertafter!(ir, v, xcall(LoggingProfiler, :record_end, fname))
        end
    end
    for (x, st) in ir
        recursable(st.expr) || continue
        ir[x] = xcall(profile_fun, st.expr.args...)
    end
    # recurse!(ir)
    return ir
end

function foo(x, y)
  z =  x * y
  z + sin(y)
end

LoggingProfiler.reset!()
@elapsed profile_fun(foo, 1.0, 1.0)
LoggingProfiler.to

macro record(ex)
    esc(Expr(:call, :profile_fun, ex.args...))
end

LoggingProfiler.reset!()
@record foo(1.0, 1.0)
LoggingProfiler.to
