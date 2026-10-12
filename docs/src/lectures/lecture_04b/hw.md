# Homework 4

::: warning Draft

Draft of the new homework 4 (2026W). Not linked from the menu.

:::

In this homework you will extend the `Diet` trait from the [lab](lab.md) by omnivores and make the
`Ecosystem.jl` package pass automated quality checks. Start from your package from the lab,
including the `Diet` trait with `Herbivore` and `Carnivore`.

## How to submit?

Zip the whole package folder `Ecosystem.jl` and upload it to BRUTE.
The package has to include at least the following files:

```
├── Project.toml
├── src
│   └── Ecosystem.jl
└── test
    ├── diet.jl      # testsets for the Diet trait
    └── runtests.jl  # includes diet.jl and runs Aqua
```

The `test/runtests.jl` file can look like this:

```julia
using Test
using Aqua
using Ecosystem

include("diet.jl")

@testset "Aqua" begin
    Aqua.test_all(Ecosystem)
end
```

## Omnivores

::: danger Homework

1. Add a new trait value `Omnivore <: Diet`. An omnivore eats
   - plants with `size > 0` (like a herbivore),
   - animals whose species is a `Herbivore` (like a carnivore).
2. Add a new species `Bear <: AnimalSpecies` with the diet `Omnivore`. Use the same constructor
   pattern as for `Sheep` and `Wolf`, e.g. `Bear(id; E=20.0, ΔE=4.0, pr=0.05, pf=0.3, s=randsex())`, and print it as 🐻.
3. Export `Omnivore` and `Bear`.
4. `eats` and `eat!` have to work for all combinations of species, following the table:

| eats ↓ \ food → | Grass | Sheep | Wolf | Bear |
| :--             | ---   | ---   | ---  | ---  |
| Sheep           | ✔     |       |      |      |
| Wolf            |       | ✔     |      |      |
| Bear            | ✔     | ✔     |      |      |

When a bear eats grass, it behaves like a sheep eating grass. When a bear eats a sheep, it behaves like a wolf eating a sheep.

**HINTS**:
- Do not modify the methods for `Herbivore` and `Carnivore`. Can you reuse them for `Omnivore`?
- Do carnivores eat bears? Check that your rule for carnivores does not need to change.

:::

## Tests and quality checks

::: danger Homework

1. Write `test/diet.jl` with testsets that check the whole table above using `eats`, and check that
   - a `Bear` eating fully grown `Grass` sets its `size` to `0`,
   - a `Bear` eating a `Sheep` removes the sheep from the `World`.
2. Add [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) as a test dependency of the package and make `Aqua.test_all(Ecosystem)` pass.

**HINTS**:
- Test dependencies go to `[extras]` and `[targets]` in `Project.toml` (or to `test/Project.toml`), see the bachelor course.
- Aqua requires `[compat]` entries for all dependencies, including julia and the test dependencies (`Aqua`, `Test`).
- Aqua checks method ambiguities. If you get one for `eats` or `eat!`, which methods are equally specific?
- `]test` has to pass in a fresh Julia session.

:::

::: details Show solution

Nothing to see here.

:::
