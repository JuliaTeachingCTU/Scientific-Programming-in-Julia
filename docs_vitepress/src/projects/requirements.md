# [Requirements](@id projects)

The goal of the project should be to create something, which is actually useful. Therefore we offer a lot of freedom in how the project will look like with the condition that you should spent around 60 hours on it (this number was derived as follows: each credit is worth 30 hours minus 13 lectures + labs minus 10 homeworks 2 hours each) and you should demonstrate some skills in solving the project. In general, we can distinguish three types of project depending on the beneficiary:

 - **You benefit:** Use / try to solve a well known problem using Julia language,
 - **Our group:** work with your tutors on a topic researched in the AIC group, 
 - **Julia community:** choose an issue in a registered Julia project you like and fix it (documentation issues are possible but the resulting documentation should be very nice.).

The project should be of sufficient complexity that verify your skill of the language (to be agreed individually).

## Project requirements

The goal of the semestral project is to create a Julia pkg with **reusable**,
**properly tested** and **documented** code. We have given you some options of topics,
as well as the freedom to choose something that could be useful for your
research or other subjects. In general we are looking for something where
performance may be crucial such as data processing, optimization or equation
solving.

In practice the project should roughly follow the structure below:

```julia
.
├── scripts
│	├── run_example.jl			# one or more examples showing the capabilities of the pkg
│	├── Project.toml 			# YOUR_PROJECT should be added here with develop command with rel path
│	└── Manifest.toml 			# should be committed as it allows to reconstruct the environment exactly
├── src
│	├── YOUR_PROJECT.jl 		# ideally only some top level code such as imports and exports, rest of the code included from other files
│	├── src1.jl 				# source files structured in some logical chunks
│	└── src2.jl
├── test
│	├── runtest.jl              # contains either all the tests or just includes them from other files
│	├── Project.toml  			# lists some additional test dependencies
│	└── Manifest.toml   		# usually not committed to git as it is generated on the fly
├── docs
│   ├── Project.toml
│   ├── make.jl
│   └── src
│       └── index.md
├── README.md 					# describes in short what the pkg does and how to install pkg (e.g. some external deps) and run the example
├── Project.toml  				# lists all the pkg dependencies
└── Manifest.toml  				# usually not committed to git as the requirements may be to restrictive
```

Make sure that 

- `README.md` is present and contains general information about the package. A small example is a nice to have.
- The package can be installed trough the package manager as `Pkg.add("url of
  the package")` with all and correct dependencies. Do not register the package
  into an official registry if you are not willing to continue its development and
  maintainance.
- Make sure that the package is covered by tests which are in the `test` folder. We
  will try to run them. There is no need for 100% percent test coverage. Tests
  testing the functionality are sufficient.
- The package should have basic documentation. For small packages, it is
  sufficient to have documentation in readme. For larger pacakges, proper
  documentation with `Documenter.jl` is advised.

Only after all this we may look at the extent of the project and it's
difficulty, which may help us in deciding between grades. 

Nice to have things, which are not strictly required but obviously improves the score.

- Ideally the project should be hosted on GitHub, which could have the continuous integration/testing set up.
- Include some benchmark and profiling code in your examples, which can show us how well you have dealt with the question of performance.
- Some parallelization attempts either by multi-processing, multi-threading, or CUDA. Do not forget to show the improvement.
- Documentation with a webpage using Documenter.jl.
