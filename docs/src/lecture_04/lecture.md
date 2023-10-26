# [Package development](@id pkg_lecture)

Organization of the code is more important with the increasing size of the project and the number of contributors and users. Moreover, it will become essential when different codebases are expected to be combined and reused. 
- Julia was designed from the beginning to encourage code reuse across different codebases as possible
- Julia ecosystem lives on a *namespace*. From then, it builds projects and environments.

## Namespaces and modules

Namespace logically separate
fragments of source code so that they can be developed independently without affecting
each other. If I define a function in one namespace, I will still be able to define another
function in a different namespace even though both functions have the same name.

- prevents confusion when common words are used in different meaning:

  - Too general name of functions "create", "extend", "loss", 
  - or data "X", "y" (especially in mathematics, think of π)
  - may not be an issue if used with different types
- *Modules* is Julia syntax for a namespace

Example:
```julia
module MySpace
function test1()
  println("test1")
end
function test2()
  println("test2")
end
export test1
#include("filename.jl")
end
```

Function `include` copies content of the file to this location (will be part of the module).

Creates functions:
```julia
MySpace.test1
MySpace.test2
```

For easier manipulation, these functions can be "exported" to be exposed to the outer world (another namespace).

Keyword: `using` exposes the exported functions and structs:
```julia
using .MySpace
```
The dot means that the module was defined in this scope.

Keyword: `import` imports function with availability to redefine it.

Combinations:

| usecase | results |
| --- | --- |
| using MySpace | MySpace.test1 |
|               | MySpace.test2 |
|               | test1 |
| using MySpace: test1              | test1 |
| import MySpace | MySpace.test1* |
|               | MySpace.test2* |
| import MySpace: test1              | test1* |
| import MySpace: test2              | test2* |

 - symbol "*" denotes functions that can be redefined

 ```julia
 using  .MySpace: test1
 test1()=println("new test")
 import  .MySpace: test1
 test1()=println("new test")
```

### Conflicts:
When importing/using functions with name that is already imported/used from another module:
- the imported functions/structs are invalidated. 
- both functions has to be acessed by their full names.

Resolution:
- It may be easier to cherry pick only the functions we need (rather than importing all via `using`)
- rename some function using keyword `as`
  ```julia
  import MySpace2: test1 as t1
  ```

### Submodules
Modules can be used or included within other modules:
```julia
module A
   a=1;
end
module B
   module C
       c = 2
   end
   b = C.c # you can read from C    (by reference)
   using ..A: a
   # a= b # but not write to A
end;
```

REPL of Julia is a module called "Main". 
- modules are not copied, but referenced, i.e. `B.b===B.C.c`
- including one module twice (from different packages) is not a problem
- Julia 1.9 has the ability to change the contextual module in the REPL:
  ```REPL.activate(TestPackage)```


### Revise.jl

The fact that Julia can redefine a function in a Module by importing it is used by package `Revise.jl` to synchronize REPL with a module or file.

So far, we have worked in REPL. If you have a file that is loaded and you want to modify it, you would need to either:
1. reload the whole file, or
2. copy the changes to REPL

`Revise.jl` does the latter automatically.

Example demo:
```julia
using Revise.jl
includet("example.jl")
```

Works with: 
- any package loaded with `import` or `using`, 
- script  loaded with `includet`, 
- Base julia itself (with Revise.track(Base))
- standard libraries (with, e.g., using Unicode; Revise.track(Unicode))

Does not work with variables!

**How it works**: monitors source code for changes and then does:
```julia
for def in setdiff(oldexprs, newexprs)
    # `def` is an expression that defines a method.
    # It was in `oldexprs`, but is no longer present in `newexprs`--delete the method.
    delete_methods_corresponding_to_defexpr(mod, def)
end
for def in setdiff(newexprs, oldexprs)
    # `def` is an expression for a new or modified method. Instantiate it.
    Core.eval(mod, def)
end
```

See [Modern Julia Workflows](https://modernjuliaworkflows.github.io) for more hints


## Namespaces & scoping

Every module introduces a new global scope. 
- Global scope
  
  - No variable or function is expected to exist  outside of it
  - Every module is equal to a global scope (no single "global" exists)
  - The REPL has a global module called `Main`

- Local scope

  Variables in Julia do not need to be explicitly declared, they are created by assignments: `x=1`. 

  In local scope, the compiler checks if variable `x` does not exist outside. We have seen:

  ```julia 
  x=1
  f(y)=x+y
  ```

  The rules for local scope determine how to treat **assignment** of `x`. If local `x` exists, it is used, if it does not:
  - in *hard* scope: new local `x` is created
  - in *soft* scope: checks if `x` exists outside (global)
    - if not: new local `x` is created
    - if yes: the split is REPL/non-interactive:
      - REPL: global `x` is used (convenience, as of 1.6)
      - non-interactive: local `x` is created


- keyword `local` and `global` can be used to specify which variable to use

From documentation:

| Construct | Scope type | Allowed within |
|:----------|:-----------|:---------------|
| `module`, `baremodule` | global | global |
| `struct` | local (soft) | global |
| `for`, `while`, `try` | local (soft) | global, local |
| `macro` | local (hard) | global |
| functions, `do` blocks, `let` blocks, comprehensions, generators | local (hard) | global, local |

Question:
```julia
x=1
f()= x=3
f()
@show x;
```

```julia
x = 1
for _ = 1:1
   x=3
end
@show x;
```

Notice that if does not introduce new scope
```julia
if true 
  branch_taken = true 
else
  branch_not_taken = true 
end
```

!!! tip "do-block"
    Let's assume a function, which takes as a first argument a function
    ```julia
      g(f::Function, args...) = println("f called on $(args) evaluates to ", f(args...))
    ```
    We can use `g` as `g(+, 1, 2)`, or with a lambda function `g(x -> x^2, 2).` But sometimes, it might be useful to the lambda function to span multiple lines. This can be achieved by a `do` block as 
    ```julia
    g(1,2,3) do a,b,c
      a*b + c
    end
    ```

## Packages

Package is a source tree with a standard layout. It provides a module and thus can be loaded with `include` or `using`.


Minimimal package:

```
PackageName/
├── src/
│   └── PackageName.jl
├── Project.toml
```

Contains:
- `Project.toml` file describing basic properties:
  - `Name`, does not have to be unique (federated package sources)
  - `UUID`, has to be unique (generated automatically)
  - optionally [deps], [targets],...

-  file `src/PackageName.jl` that defines module `PackageName` which is executed when loaded.


Many other optional directories:
- directory tests/,  (almost mandatory)
- directory docs/    (common)
- directory scripts/, examples/,... (optional)


!!! tip "Type-Piracy"
    "Type piracy" refers to the practice of extending or redefining methods in Base or other packages on types that you have not defined. In extreme cases, you can crash Julia (e.g. if your method extension or redefinition causes invalid input to be passed to a ccall). Type piracy can complicate reasoning about code, and may introduce incompatibilities that are hard to predict and diagnose.
    ```julia
    module A
    import Base.*
    *(x::Symbol, y::Symbol) = Symbol(x,y)
    end
    ```
  
The package typically loads other modules that form package dependencies.

## Project environments

Is a package that does not contain `Name` and `UUID` in `Project.toml`. It's used when you don't need to create a package for your work. It's created by `activate some/path` in REPL package mode. 

## Project Manifest

Both package and environment can contain an additional file `Manifest.toml`.
This file tracks full dependency tree of a project including versions of the packages on which it depends.

for example:
```toml
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"
```

Content of files `Project.toml` and `Manifest.toml` are maintained by PackageManager.

## Package manager

Handles both packages and projects:
- creating a project `]generate PkgName`
- adding an existing project `add PkgName` or ` add https://github.com/JuliaLang/Example.jl`

    Names are resolved by Registrators (public or private).

- removing `]rm PkgName`
- updating `]update`
- developing `]dev http://...` 
  - `add` treats packages as being finished, version handling pkg manager. Precompiles!
  - `dev` leaves all operations on the package to the user (git versioning, etc.). Always read content of files

By default these operations are related to environment `.julia/environments/v1.9`

E.g. running and updating will update packages in `Manifest.toml` in this directory. What if the update breaks functionality of some project package that uses special features?

There can and *should* be more than one environment!

Project environments are based on files with installed packages.

- switching by `]activate Path` - creates `Project.toml` if not existing

- from that moment, all package modifications will be relevant only to this project!
- when switching to a new project `]instantiate` will prepare (download and precompile) the environment
  - creates `Manifest.toml` = list of all exact versions of all packages 
- which Packages are visible is determined by `LOAD_PATH`
    - typically contaings default libraries and default environment
    - it is different for REPL and Pkg.tests ! No default env. in tests. 


## Package hygiene - workflow

!!! theorem "Potential danger"
    Package dependencies may not be compatible: 
    - package `A` requires `C@<0.2`
    - package `B` requires `C@>0.3`
    - what should happen when `]add A` and `add B`?

- keep your "@v#.#" as clean as possible (recommended are only debugging/profiling packages)
- use packages as much as you can, even for short work with scripts `]activate .`
  - adding a package existing elsewhere is cheap (global cache)
- if do you not wish to store any files just test random tricks of a cool package: `]activate --temp`

### Package development with Revise

Developing a package with interactive test/development:

1. Create a package/module at one directory `MainPackage`
2. Create a script at another directory `MainScript`, and activate it `]activate .`
3. `dev MainPackage` in the `MainScript` environment
   - Revise.jl will watch the `MainPackage` so it is always up to date
   - in `dev` mode you have full control over commits etc.


### Package Extensions

Some functionality of a package that depends on external packages may not be always needed. A typical example is plotting and visualization that may reguire heavy visualization packages.
These are completely unnecessary e.g. in distributed server number crunching.

The extension is a new module depending on: i) the base package, and ii) the conditioning package.
It will not be compiled if the conditioning package is not loaded. Once the optional package is loaded, the extension will be automatically compiled and loaded.

New feature since Julia 1.9:
- new directory in project tree: `ext/` the extending module is stored here
- new section in `Project.toml` called `[extensions]` listing extension names and their conditioning packages





## Unit testing, /test

Without explicit keywords for checking constructs (think missing functions in interfaces), the good quality of the code is guaranteed by detailed unit testing.

- each package should have directory `/test`
- file `/test/runtest.jl` is run by the command `]test` of the package manager

  this file typically contains `include` of other tests

- no formal structure of tests is prescribed
  - test files are just ordinary julia scripts
  - user is free to choose what to test and how (freedom x formal rules)

- testing functionality is supported by macros `@test` and `@teststet`

  ```
  @testset "trigonometric identities" begin
      θ = 2/3*π
      @test sin(-θ) ≈ -sin(θ)
      @test cos(-θ) ≈ cos(θ)
      @test sin(2θ) ≈ 2*sin(θ)*cos(θ)
      @test cos(2θ) ≈ cos(θ)^2 - sin(θ)^2
  end;
  ```  

Testset is a collection of tests that will be run and summarized in a common report.
  - Testsets can be nested: testsets in testsets
  - tests can be in loops or functions

    ```julia
    for i=1:10
       @test a[i]>0
    end
    ```

  - Useful macro `≈` checks for equality with given tolerance

    ```julia
    a=5+1e-8
    @test a≈5
    @test a≈5 atol=1e-10
    ```    
  
- @testset resets RNG to Random.GLOBAL_SEED before and after the test for repeatability 

  The same results of RNG are not guaranteed between Julia versions!

- Test coverage: package `Coverage.jl`
- Can be run automatically by continuous integration, e.g. GitHub actions
- integration in VSCode test via package `TestItems.jl` 


## Documentation & Style, /docs

A well written package is reusable if it is well documented. 

The simpliest kind of documentation is the docstring:
```julia
"Auxiliary function for printing a hello"
hello()=println("hello")

"""
More complex function that adds π to input:
- x is the input argument (itemize)

Can be written in latex: ``x \leftarrow x + \pi``
"""
addπ(x) = x+π
```

Yieds:

!!! tip "Renders as"
    More complex function that adds π to input:
    - x is the input argument (itemize)

    Can be written in latex: ``x \leftarrow x + \pi``

Structure of the document
```
PackageName/
├── src/
│   └── SourceFile.jl
├── docs/
│    ├── build/
│    ├── src/
│    └── make.jl
...
```

Where the line-by-line documentation is in the source files.
- `/docs/src` folder can contain more detailed information: introductory pages, howtos, tutorials, examples
- running `make.jl` controls which pages are generated in what form (html or latex) documentation in the /build  directory
- automated with GitHub actions

Documentation is generated by the julia code.
- code in documentation can be evaluated
  
```@repl test
x=3
@show x
```
    
- documentation can be added by code:

  ```julia
  struct MyType
      value::String
  end
  
  Docs.getdoc(t::MyType) = "Documentation for MyType with value $(t.value)"

  x = MyType("x")
  y = MyType("y")
  ```

  See `?x` and `?y`. 
    
  It uses the same very standard building blocks: multiple dispatch.

## Precompilation

By default, every package is precompiled when loading and stored in compiled form in a cache.

If it defines methods that extend previously defined (e.g. from Base), it may affect already loaded packages which need to be recompiled as well. May take time.

Julia has a tracking mechanism that stores information about the whole graph of dependencies. 

Faster code can be achieved by the `precompile` directive:
```julia
module FSum

fsum(x) = x
fsum(x,p...) = x+fsum(p...)

precompile(fsum,(Float64,Float64,Float64))
end
```

Can be investigated using `MethodAnalysis`.

```julia
using MethodAnalysis
mi =methodinstances(fsum)
```

Useful packages:
- `PackageCompiler.jl` has three main purposes:

  - Creating custom sysimages for reduced latency when working locally with packages that has a high startup time.
  - Creating "apps" which are a bundle of files including an executable that can be sent and run on other machines without Julia being installed on that machine.
  - Creating a relocatable C library bundle form of Julia code.



- `AutoSysimages.jl` allows easy generation of precompiles images - reduces package loading

## Additional material
- [Modern Julia Workflows](https://modernjuliaworkflows.github.io)
