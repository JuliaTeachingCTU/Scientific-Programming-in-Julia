# A simple reverse-mode AD.
# Lots of simplifications have been made (in particular, there is no support for
# control flow). But this illustrates most of the principles behind Zygote.
# https://fluxml.ai/Zygote.jl/dev/internals/


#####

# We assume to have a set of AD rules (e.g. ChainRules), which for a given function returns its evaluation and pullback. If we are tasked with computing the gradient.

# 1. If a rule exists for this function, directly return the rule.
# 2. If not, deconstruct the function into a sequence of functions by asking f `IRCode`
# 3. Replace statements by calls to obtain the evaluation of the statements and the pullback.
# 4. Chain pullbacks in reverse order.
# 5. Return the function evaluation and the chained pullback.

# The idea is that we will replace each statement of `foo` with a statement returning the function value and pullback. At the moment and for simplicity, we assume that appropriate chain is defined. Moreover, we need to keep track of mapping old SSAValues to new SSAValues in ssamap, since their values will differ.

import Core.Compiler as CC
using ChainRules
using Core: SSAValue, GlobalRef, ReturnNode


function get_ciir(f, sig; world = Core.Compiler.get_world_counter(), optimize_until = "compact 1")
  mi = only(Base.method_instances(f, sig, world))
  ci = Base.uncompressed_ir(mi.def::Method)
  (ir, rt) = only(Base.code_ircode(f, sig; optimize_until))
  (copy(ci), ir, rt)
end



# struct Pullback{S,T}
#   pullbacks::T
# end


argtype(ir::CC.IRCode, a::Core.Argument) = ir.argtypes[a.n]
argtype(ir::CC.IRCode, a::Core.SSAValue) = ir.stmts.type[a.id]
argtype(ir::CC.IRCode, f::GlobalRef) = typeof(eval(f))
argtype(ir::CC.IRCode, a) = error("argtype of $(typeof(a)) not supported")

"""
  type_of_pullback(ir, inst)

  infer type of the pullback
"""
function type_of_pullback(ir, inst, optimize_until = "compact 1")
  inst.head != :call && error("inferrin return type of calls is supported")
  params = tuple([argtype(ir, a) for a in inst.args]...)
  (ir, rt) = only(Base.code_ircode(ChainRules.rrule, params; optimize_until))
  if !(rt <:Tuple{A,B} where {A,B})
    error("The return type of pullback `ChainRules.rrule($(params))` should be tuple")
  end
  rt
end

remap(d, args::Tuple) = map(a -> remap(d,a), args) 
remap(d, args::Vector) = map(a -> remap(d,a), args) 
remap(d, r::ReturnNode) = ReturnNode(remap(d, r.val))
remap(d, x::SSAValue) = d[x] 
remap(d, x) = x

function forward(ir)
  pullbacks = []
  new_insts = Any[]
  new_line = Int32[]
  new_types = Any[]
  ssamap = Dict{SSAValue,SSAValue}()
  fval_ssa = nothing
  for (i, stmt) in enumerate(ir.stmts)
    inst = stmt[:inst]
    if inst isa Expr && inst.head == :call
      new_inst = Expr(:call, GlobalRef(ChainRules, :rrule), remap(ssamap, inst.args)...)
      tt = type_of_pullback(ir, inst)
      push!(new_insts, new_inst)
      push!(new_line, stmt[:line])
      push!(new_types, tt)
      rrule_ssa = SSAValue(length(new_insts))


      push!(new_insts, Expr(:call, :getindex, rrule_ssa, 1))      
      push!(new_line, stmt[:line])
      push!(new_types, tt.parameters[1])
      val_ssa = SSAValue(length(new_insts))
      ssamap[SSAValue(i)] = val_ssa
      (stmt[:type] != tt.parameters[1]) && @info("pullback of $(inst) has a different type than normal inst")

      push!(new_insts, Expr(:call, :getindex, rrule_ssa, 2))
      pullback_ssa = SSAValue(length(new_insts))
      push!(new_line, stmt[:line])
      push!(new_types, tt.parameters[2])
      push!(pullbacks, pullback_ssa)
      continue
    end

    if inst isa ReturnNode
      fval_ssa = remap(ssamap, inst.val)
      continue
    end
    error("unknown node $(i)")
  end

  # construct tuple with all pullbacks
  push!(new_insts, Expr(:call, :tuple, pullbacks...))
  pull_ssa = SSAValue(length(new_insts))
  push!(new_line, new_line[end])
  push!(new_types, Tuple{[new_types[x.id] for x in pullbacks]...})

  # construct the tuple containing forward and reverse
  push!(new_insts, Expr(:call, :tuple, fval_ssa, pull_ssa))
  ret_ssa = SSAValue(length(new_insts))
  push!(new_line, new_line[end])
  push!(new_types, Tuple{new_types[fval_ssa.id], new_types[pull_ssa.id]})

  # put a return statement
  push!(new_insts, ReturnNode(ret_ssa))
  push!(new_line, new_line[end])
  push!(new_types, Any)

  # this nightmare construct the IRCode with absolutely useless type information
  is = CC.InstructionStream(
    new_insts,                               # inst::Vector{Any}
    new_types,            # type::Vector{Any}
    fill(CC.NoCallInfo(), length(new_insts)),   # info::Vector{CallInfo}
    new_line,                               # line::Vector{Int32}
    fill(UInt8(0), length(new_insts)),       # flag::Vector{UInt8}
  )
  cfg = CC.compute_basic_blocks(new_insts)
  new_ir = CC.IRCode(is, cfg, ir.linetable, ir.argtypes, ir.meta, ir.sptypes)
end





function foo(x,y) 
  z = x * y 
  z + sin(x)
end


(ci, ir, rt) = get_ciir(foo, (Float64, Float64))
new_ir = forward(ir)
CC.replace_code_newstyle!(ci, ir)

forw = Core.OpaqueClosure(forward(ir))
fval, pullbacks = forw(1.0,1.0)

(1.0,1.0)


"""
  function reverse(ir)

  we construct the reverse using the original `ir` code, since we can obtain it in from the 
  parameter `S` of the Pullback{S,T}. `S` can contain `(foo, Float64,Float64)` when 
  we compute the gradient of `foo`.  
"""
function reverse(ir)
  diffmap = Dict{Any,Any}() # this will hold the mapping where is the gradient with respect to SSA.
  # the argument of the pullback we are defining is a gradient with respect to the argument of return
  # which we assume to be the last of insturction in `inst`
  @assert ir.stmts.inst[end] isa ReturnNode
  diffmap[ir.stmts.inst[end].val] = Core.Argument(2)

  reverse_inst = []

  # a first IR will be to get a structure will pullbacks from the first argument 
  push!(reverse_inst, Expr(:call, GlobalRef(Core, :getfield), Core.Argument(1), :pullbacks))
  pullbacks_ssa = SSAValue(length(reverse_inst))

  # we should filter statements from which we can get the pullbacks, but for the trivial
  # function without control flow this is not needed

  # now we iterate over pullbacks and execute one by one with correct argument
  for i in (length(ir.stmts)-1):-1:1
    inst = ir.stmts[i][:inst]
    val_ssa = SSAValue(i)

    #first, we get the pullback
    push!(reverse_inst, Expr(:call, GlobalRef(Base, :getindex), pullbacks_ssa, i))
    pullback_ssa = SSAValue(length(reverse_inst))

    #execute pullback
    push!(reverse_inst, Expr(:call, pullback_ssa, diffmap[val_ssa]))
    arg_grad = SSAValue(length(reverse_inst))
    for (j, a) in enumerate(inst.args)
      j == 1 && continue # we omit gradient with respect to the name of the function and rrule
      if haskey(diffmap, a)  # we need to perform addition
        push!(reverse_inst, Expr(:call, GlobalRef(Base, :getindex), arg_grad, j))
        sv = SSAValue(length(reverse_inst))
        push!(reverse_inst, Expr(:call, GlobalRef(Base, :+), sv, diffmap[a]))
        diffmap[a] = SSAValue(length(reverse_inst))
      else
        push!(reverse_inst, Expr(:call, GlobalRef(Base, :getindex), arg_grad, j))
        diffmap[a] = SSAValue(length(reverse_inst))
      end
    end
  end

end
