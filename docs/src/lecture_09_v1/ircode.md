These notes are from poking around `Core.Compiler` to see, how they are different from working just with `CodeInfo` and `IRTools.jl`. Notes are mainly around IRCode. Why there is a `Core.Compiler.IRCode` when there was `Core.CodeInfo`? Seems to be historical reasons. At the beginning, Julia did not have any intermediate representation and code directly emitted LLVM. Then, it has received an `CodeInfo` as in intermediate representation. `IRCode` seems like an evolution of `CodeInfo`. `Core.Compiler` works mostly with `IRCode`, but the `IRCode`  can be converted to the `CodeInfo` and the other way around. `IRCode` seems to be designed more for implementation of various optimisation phases. Personal experience tells me it is much nicer to work with even on the low level. 

Throughout the explanation, we assume that `Core.Compiler` was imported as `CC` to decrease the typing load.

Let's play with a simple silly function 
```julia

function foo(x,y) 
  z = x * y 
  z + sin(x)
end
```

### IRCode
We can obtain `CC.IRCode`
```julia
import Core.Compiler as CC
(ir, rt) = only(Base.code_ircode(foo, (Float64, Float64), optimize_until = "compact 1"))
```
which returns `Core.Compiler.IRCode` in `ir` and return-type `Float64` in `rt`.
The output might look like 
```
julia> (ir, rt) = only(Base.code_ircode(foo, (Float64, Float64), optimize_until = "compact 1"))
  1─ %1 = (_2 * _3)::Float64                                                    
  │   %2 = Main.sin(_2)::Float64                                                 
  │   %3 = (%1 + %2)::Float64                                                    
  └──      return %3                                                             
   => Float64

```
Options of `optimize_until` are `compact 1`, `compact 2`, `nothing.` I do not see a difference between `compact 2` and `compact 2`.

The IRCode structure is defined as
```
struct IRCode
    stmts::InstructionStream
    argtypes::Vector{Any}
    sptypes::Vector{VarState}
    linetable::Vector{LineInfoNode}
    cfg::CFG
    new_nodes::NewNodeStream
    meta::Vector{Expr}
end
```
where
* `stmts` is a stream of instruction (more in this below)
* `argtypes` holds types of arguments of the function whose `IRCode` we have obtained
* `sptypes` is a vector of `VarState`. It seems to be related to parameters of types
* `linetable` is a table of unique lines in the source code from which statements 
* `cfg` holds control flow graph, which contains building blocks and jumps between them
* `new_nodes` is an infrastructure that can be used to insert new instructions to the existing `IRCode` . The idea behind is that since insertion requires a renumbering all statements, they are put in a separate queue. They are put to correct position with a correct `SSANumber`  by calling `compact!`.
* `meta` is something.

Before going further, let's take a look on `InstructionStream` defined as 
```julia
struct InstructionStream
    inst::Vector{Any}
    type::Vector{Any}
    info::Vector{CallInfo}
    line::Vector{Int32}
    flag::Vector{UInt8}
end
```
where 
* `inst` is a vector of instructions, stored as `Expr`essions. The allowed fields in `head` are described [here](https://docs.julialang.org/en/v1/devdocs/ast/#Expr-types)
* `type` is the type of the value returned by the corresponding statement
* `CallInfo` is ???some info???
* `line` is an index into `IRCode.linetable` identifying from which line in source code the statement comes from
* `flag`  are some flags providing additional information about the statement.
	- `0x01 << 0` = statement is marked as `@inbounds`
	- `0x01 << 1` = statement is marked as `@inline`
	- `0x01 << 2` = statement is marked as `@noinline`
	- `0x01 << 3` = statement is within a block that leads to `throw` call
	- `0x01` << 4 = statement may be removed if its result is unused, in particular it is thus be both pure and effect free
	- `0x01 << 5-6 = <unused>`
	- `0x01 << 7 = <reserved>` has out-of-band info

For the above `foo` function, the InstructionStream looks like

```julia
julia> DataFrame(flag = ir.stmts.flag, info = ir.stmts.info, inst = ir.stmts.inst, line = ir.stmts.line, type = ir.stmts.type)
4×5 DataFrame
 Row │ flag   info                               inst          line   type
     │ UInt8  CallInfo                           Any           Int32  Any
─────┼────────────────────────────────────────────────────────────────────────
   1 │   112  MethodMatchInfo(MethodLookupResu…  _2 * _3           1  Float64
   2 │    80  MethodMatchInfo(MethodLookupResu…  Main.sin(_2)      2  Float64
   3 │   112  MethodMatchInfo(MethodLookupResu…  %1 + %2           2  Float64
   4 │     0  NoCallInfo()                       return %3         2  Any
```

We can index into the statements as `ir.stmts[1]`, which provides a "view" into the vector. To obtain the first instruction, we can do `ir.stmts[1][:inst]`.

The IRCode is typed, but the fields can contain `Any`. It is up to the user to provide corrrect types of the output and there is no helper functions to perform typing. A workaround is shown in the Petite Diffractor project. Julia's sections of the manual https://docs.julialang.org/en/v1/devdocs/ssair/ and seems incredibly useful. The IR form they talk about seems to be `Core.Compiler.IRCode`. 

It seems to be that it is possible to insert IR instructions into the it structure by queuing that to the field `stmts` and then call `compact!`, which would perform the heavy machinery of relabeling everything.

#### Example of modifying the function through IRCode
Below is an MWE that tries to modify the IRCode of a function and execute it.  The goal is to change the function `foo`  to `fooled`.

```julia
import Core.Compiler as CC
using Core: SSAValue, GlobalRef, ReturnNode

function foo(x,y) 
  z = x * y 
  z + sin(x)
end

function fooled(x,y) 
  z = x * y 
  z + sin(x) + cos(y)
end

(ir, rt) = only(Base.code_ircode(foo, (Float64, Float64), optimize_until = "compact 1"));
nr = CC.insert_node!(ir, 2, CC.NewInstruction(Expr(:call, Core.GlobalRef(Main, :cos), Core.Argument(3)), Float64))
nr2 = CC.insert_node!(ir, 4,  CC.NewInstruction(Expr(:call, GlobalRef(Main, :+), SSAValue(3), nr), Float64))
CC.setindex!(ir.stmts[4],  ReturnNode(nr2), :inst)
ir = CC.compact!(ir)
irfooled = Core.OpaqueClosure(ir)
irfooled(1.0, 2.0) == fooled(1.0, 2.0)
```

So what we did?
1. `(ir, rt) = only(Base.code_ircode(foo, (Float64, Float64), optimize_until = "compact 1"))` obtain the `IRCode` of the function `foo` when called with both arguments being `Float64`. `rt` contains the return type of the 
2. A new instruction `cos` is inserted to the `ir` by  `Core.Compiler.insert_node!`, which takes as an argument an `IRCode`, position (2 in our case), and new instruction. The new instruction is created by `NewInstruction` accepting as an input expression `Expr` and a return type. Here, we force it to be `Float64`, but ideally it should be inferred. (This would be the next stage). Or, may-be, we can run it through type inference? . The new instruction is added to the `ir.new_nodes` instruction stream and obtain a new SSAValue returned in `nr`, which can be then used further.
3. We add one more instruction `+` that uses output of the instruction we add in step 2, `nr` and SSAValue from statement 3 of the original IR (at this moment, the IR is still numbered with respect to the old IR, the renumbering will happen later.)  The output of this second instruction is returned in `nr2`.
4. Then, we rewrite the return statement to return `nr2` instead of `SSAValue(3)`.
5. `ir = CC.compact!(ir)` is superimportant since it moves the newly added statements from `ir.new_stmts` to `ir.stmts` and importantly renumbers `SSAValues.` *Even though the function is mutating, the mutation here is meant that the argument is changed, but the new correct IRCode is returned and therefore has to be reassigned.*
6. The function is created through `OpaqueClosure.`
7. The last line certifies that the function do what it should do.

There is no infrastructure to make the above manipulation transparent, like is the case of @generated function and codeinfo. It is possible to hook through generated function by converting the IRCode to untyped CodeInfo, in which case you do not have to bother with typing.

#### How to obtain code info the proper way?
This is the way code info is obtained in the diffractor.
```julia
mthds = Base._methods_by_ftype(sig, -1, world)
match = only(mthds)

mi = Core.Compiler.specialize_method(match)
ci = Core.Compiler.retrieve_code_info(mi, world)
```



### CodeInfo



`IRTools.jl` are great for modifying `CodeInfo`. I have found two tools for modifying `IRCode` and  I wonder if they have been abandoned because they were both dead ends or because of lack of human labor. I am also aware of Also, [this](https://nbviewer.org/gist/tkf/d4734be24d2694a3afd669f8f50e6b0f/00_notebook.ipynb) is quite cool play with IRStuff.


Resources
* https://vchuravy.dev/talks/licm/
* [CompilerPluginTools](https://github.com/JuliaCompilerPlugins/CompilerPluginTools.jl) 
* [CodeInfoTools.jl](https://github.com/JuliaCompilerPlugins/CodeInfoTools.jl).
*  TKF's [CodeInfo.jl](https://github.com/tkf/ShowCode.jl) is nice for visualization of the IRCode
* Diffractor is an awesome source of howto. For example function `my_insert_node!` in `src/stage1/hacks.jl`
* https://nbviewer.org/gist/tkf/d4734be24d2694a3afd669f8f50e6b0f/00_notebook.ipynb
* https://github.com/JuliaCompilerPlugins/Mixtape.jl

