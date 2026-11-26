using MacroTools
using IRTools

using IRTools: branches, block, empty!, evalir, func, branch!, block, IR, @dynamo, xcall

function _mark(label, ex)
    label isa Symbol || error("label has to be a Symbol")
    return Expr(
        :block,
        Expr(:meta, :begin_optional, label),
        esc(ex),
        Expr(:meta, :end_optional, label),
    )
end

macro mark(label, ex)
    _mark(label, ex)
end


foo(x) = bar(baz(x))

function bar(x)
    @mark print iseven(x) && println("The input is even.")
    x
end

function baz(x)
    @mark print x<0 && println("The input is negative.")
    x
end



isbegin(e::Expr) = Meta.isexpr(e,:meta) && e.args[1]===:begin_optional
isend(e::Expr) = Meta.isexpr(e,:meta) && e.args[1]===:end_optional


skip(f::Core.IntrinsicFunction, args...) = f(args...)
skip(f::Core.Builtin, args...) = f(args...)

@dynamo function skip(args...)
    ir = IR(args...)
    delete_line = false
    local orig

    for (x,st) in ir
        is_begin = isbegin(st.expr)
        is_end   = isend(st.expr)

        if is_begin
            delete_line = true
        end
    
        if is_begin
            orig = block(ir,x)
        elseif is_end
            dest = block(ir,x)
            if orig != dest
                empty!(branches(orig))
                branch!(orig,dest)
            end
        end

        if delete_line
            delete!(ir,x)
        end

        if is_end
            delete_line = false
        end

        if haskey(ir,x) && Meta.isexpr(st.expr,:call)
            ir[x] = IRTools.xcall(skip, st.expr.args...)
        end
    end
    return ir
end

function skip(ex::Expr)
end

macro skip(ex)
    ex.head == :call || error("Input expression has to be a `:call`.")
    return xcall(skip, ex.args...)
end

display(@code_ir foo(-2))
display(@code_ir skip(foo,-2))
display(foo(-2))
@skip foo(-2)
