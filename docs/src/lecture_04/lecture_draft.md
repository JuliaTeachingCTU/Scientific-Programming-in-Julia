# Package development

## namespaces & scoping
- local/global scope
- soft/hard
- let 
- for loops?
  - (outer keyword)
- differences in REPL?

## modules
- exporting
- conflicts

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
- pitfals testing (checkbounds = yes)
- testing in github-actions
- testing with RNG (version changes)
- testing gradients (FiniteDifferences)

## Precompilation
- new method -> invalidating methods
- cache