# Package development


## Modules, namespaces
- why namespaces
  - Too general name "create", "extend", "loss", 
  - may not be an issue if used with different types
- Julia syntax
  ```julia
  module MySpace
  function test1
    println("test1")
  end
  function test1
    println("test1")
  end
  end
  ```

- using .MySpace
- export, import 
- conflicts (as )

## namespaces & scoping
- local/global scope
- soft/hard

- table of soft vs. hard


## packages = modules
- template structure src/, script/, tests/, docs/
- dependencies (how to resolve?)
- sub-modules
- cyclic dependencies
- dev packages

## Package manager
- repeatability
- compat function
- registrator (available to others) (also localy!)
- environments, (activate)

- using, import
- interfaces (Base.+, Stempty functions)

## Unit testing
- write tests
  - check e.g. interfaces (i.e. missing functions)
  - @test, @testset
- pitfals testing (checkbounds = yes)?
- testing in github-actions
- testing with RNG (version changes)
- testing gradients (FiniteDifferences)

## Precompilation
- new method -> invalidating methods
- cache

## Revise.jl
- load before others
- includet for files
- adding deleting functions from module works

## Style Guide
- docstrings
- style
  - modules, types - CamelCase
  - functions, instances - lowercase
  - modifying function !
