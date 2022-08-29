# [Projects](@id projects)

We want you to use Julia for something that is acutally useful for you.
Therefore you can choose your final, graded project very freely.
We will discuss your ideas with you individually and come up with a sufficiently
extensive project together.

In general, we can distinguish project depending on the beneficiary:
 - You: try new language for a problem well known to you,
 - Our group: wort with your tutors on a topic researched in the AIC group 
 - Julia community: choose an issue in a registered Julia project you like and fix it (including documentation issues)

 The project should be of sufficient complexity that verify your skill of the language (to be agreed individually)


For you inspiration of what such a project could look like we have four
suggestions for you (which you can of course choose to work on as well).

## The Equation Learner And Its Symbolic Representation

In many scientific and engineering one searches for interpretable (i.e.
human-understandable) models instead of the black-box function approximators
that neural networks provide.
The [*equation learner*](http://proceedings.mlr.press/v80/sahoo18a.html) (EQL)
is one approach that can identify concise equations that describe a given
dataset.

The EQL is essentially a neural network with different unary or binary
activation functions at each indiviual unit. The network weights are
regularized during training to obtain a sparse model which hopefully results in
a model that represents a simple equation.

The goal of this project is to implement the EQL, and if there is enough time
the [*improved equation learner*](https://arxiv.org/abs/2105.06331) (iEQL).
The equation learners should be tested on a few toy problems (possibly inspired
by the tasks in the papers).  Finally, you will implement functionality that
can transform the learned model into a symbolic, human readable, and exectuable
Julia expression.

## An Evolutionary Algorithm Applied To Julia's AST

Most of the approaches to equation learning have to be differentiable by default
in order to use the traditional machinery of stochastic gradient descent with
backpropagation. This often leads to equations with too many terms, requiring 
special techniques for enforcing sparsity for terms with low weights.

In Julia we can however use a different learning paradigm of evolutionary 
algorithms, which can work on discrete set of expressions. The goal is to 
write mutation and recombination - the basic operators of a genetic algorithm,
but applied on top of Julia AST.

Data: AI Feyman [database](https://space.mit.edu/home/tegmark/aifeynman.html) on symbolic regression (from [article](https://arxiv.org/pdf/1905.11481.pdf)/[code](https://github.com/SJ001/AI-Feynman))
Inspiration: 
- Logic Guided Genetic Algorithms [article](https://arxiv.org/pdf/2010.11328.pdf)/[code](https://github.com/DhananjayAshok/LGGA)
- AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity [article](https://arxiv.org/pdf/2006.10782)
- Genetic Programming for Julia: fast performance and parallel island model implementation [report](http://courses.csail.mit.edu/18.337/2015/projects/MorganFrank/projectReport.pdf)

## Distributed Optimization Package

One click distributed optimization is at the heart of other machine learning 
and optimization libraries such as pytorch, however some equivalents are 
missing in the Julia's Flux ecosystem. The goal of this project is to explore,
implement and compare at least two state-of-the-art methods of distributed 
gradient descent on data that will be provided for you.

Some of the work has already been done in this area by one of our former students, 
see [link](https://dspace.cvut.cz/handle/10467/97057).

## A Rule Learning Algorithm

[Rule-based models](https://christophm.github.io/interpretable-ml-book/rules.html)
are simple and very interpretable models that have been around for a long time
and are gaining popularity again.
The goal of this project is to implement a
[sequential covering](https://christophm.github.io/interpretable-ml-book/rules.html#sequential-covering)
algorithm called [`RIPPER`](http://www.cs.utsa.edu/~bylander/cs6243/cohen95ripper.pdf)
and evaluate it on a number of datasets.


# Project requirements
The goal of the semestral project is to create a Julia pkg with **reusable, properly tested and documented** code. We have given you some options of topics, as well as the freedom to choose something that could be useful for your research or other subjects. In general we are looking for something where performance may be crucial such as data processing, optimization or equation solving.

In practice the project should follow roughly this tree structure
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
├── README.md 					# describes in short what the pkg does and how to install pkg (e.g. some external deps) and run the example
├── Project.toml  				# lists all the pkg dependencies
└── Manifest.toml  				# usually not committed to git as the requirements may be to restrictive
```

The first thing that we will look at is `README.md`, which should warn us if there are some special installation steps, that cannot be handled with Julia's Pkg system. For example if some 3rd party binary dependency with license is required. Secondly we will try to run tests in the `test` folder, which should run and not fail and should cover at least some functionality of the pkg. Thirdly and most importantly we will instantiate environment in `scripts` and test if the example runs correctly. Lastly we will focus on documentation in terms of code readability, docstrings and inline comments. 

Only after all this we may look at the extent of the project and it's difficulty, which may help us in deciding between grades. 

Nice to have things, which are not strictly required but obviously improves the score.
- Ideally the project should be hosted on GitHub, which could have the continuous integration/testing set up.
- Include some benchmark and profiling code in your examples, which can show us how well you have dealt with the question of performance.
- Some parallelization attempts either by multi-processing, multi-threadding, or CUDA. Do not forget to show the improvement.
- Documentation with a webpage using Documenter.jl.

Here are some examples of how the project could look like:

- [ImageInspector](https://github.com/JuliaTeachingCTU/ImageInspector.jl)
