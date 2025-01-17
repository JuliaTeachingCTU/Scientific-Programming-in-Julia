# Potential projects

Below, we list some potential projects for inspiration.

## Implementing new things

### Lenia (Continuous Game of Life)

[Lenia](https://chakazul.github.io/lenia.html#Code) is a continuous version of Conway's Game of
Life. Implement a Julia version. For example, you could focus either on performance compared to the
python version, or build nice visualizations with [Makie.jl](https://docs.makie.org/stable/).

Nice tutorial [from Conway to Lenia](https://colab.research.google.com/github/OpenLenia/Lenia-Tutorial/blob/main/Tutorial_From_Conway_to_Lenia.ipynb)

### The Equation Learner And Its Symbolic Representation

In many scientific and engineering one searches for interpretable (i.e.
human-understandable) models instead of the black-box function approximators
that neural networks provide.
The [*equation learner*](http://proceedings.mlr.press/v80/sahoo18a.html) (EQL)
is one approach that can identify concise equations that describe a given
dataset.

The EQL is essentially a neural network with different unary or binary
activation functions at each individual unit. The network weights are
regularized during training to obtain a sparse model which hopefully results in
a model that represents a simple equation.

The goal of this project is to implement the EQL, and if there is enough time
the [*improved equation learner*](https://arxiv.org/abs/2105.06331) (iEQL).
The equation learners should be tested on a few toy problems (possibly inspired
by the tasks in the papers).  Finally, you will implement functionality that
can transform the learned model into a symbolic, human readable, and executable
Julia expression.

### Architecture visualizer

Create an extension of Flux / Lux and to visualize architecture of a neural network suitable for publication. Something akin [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet).

### Learning Large Language Models with reduced precition (Mentor: Tomas Pevny)

Large Language Models ((Chat) GPT, LLama, Falcon, Palm, ...) are huge. A recent trend is to perform optimization in reduced precision, for example in `int8` instead of `Float32`. Such feature is currently missing in Julia ecosystem and this project should be about bringing this to the community (for an introduction, read these blogs [*LLM-int8 and emergent features*](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/), [*A gentle introduction to 8-bit Matrix Multiplication*](https://huggingface.co/blog/hf-bitsandbytes-integration)). The goal would be to implement this as an additional type of Number / Matrix and overload multiplication on CPU (and ideally on GPU) to make it transparent for neural networks? **What I will learn?** In this project, you will learn a lot about the (simplicity of) implementation of deep learning libraries and you will practice abstraction of Julia's types. You can furthermore learn about GPU Kernel programming and `Transformers.jl` library.

### Planning algorithms (Mentor: Tomas Pevny)

Extend [SymbolicPlanners.jl](https://github.com/JuliaPlanners/SymbolicPlanners.jl) with the mm-ϵ variant of the bi-directional search [MM: A bidirectional search algorithm that is guaranteed to meet in the middle](https://www.sciencedirect.com/science/article/pii/S0004370217300905). This [pull request](https://github.com/JuliaPlanners/SymbolicPlanners.jl/pull/8) might be very helpful in understanding better the library.

### A Rule Learning Algorithms (Mentor: Tomas Pevny)

[Rule-based models](https://christophm.github.io/interpretable-ml-book/rules.html)
are simple and very interpretable models that have been around for a long time
and are gaining popularity again.
The goal of this project is to implement one of these algorithms
* [sequential covering](https://christophm.github.io/interpretable-ml-book/rules.html#sequential-covering)
algorithm called [`RIPPER`](http://www.cs.utsa.edu/~bylander/cs6243/cohen95ripper.pdf)
and evaluate it on a number of datasets.
* [Learning Certifiably Optimal Rule Lists for Categorical Data](https://arxiv.org/abs/1704.01701)
* [Boolean decision rules via column generation](https://proceedings.neurips.cc/paper/2018/file/743394beff4b1282ba735e5e3723ed74-Paper.pdf)
* [Learning Optimal Decision Trees with SAT](https://proceedings.neurips.cc/paper/2021/file/4e246a381baf2ce038b3b0f82c7d6fb4-Paper.pdf)
* [A SAT-based approach to learn explainable decision sets](https://link.springer.com/content/pdf/10.1007/978-3-319-94205-6_41.pdf)
To increase the impact of the project, consider interfacing it with [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/)

### Parallel optimization (Mentor: Tomas Pevny)

Implement one of the following algorithms to train neural networks in parallel. Can be implemented in a separate package or consider extending [FluxDistributed.jl](https://github.com/DhairyaLGandhi/FluxDistributed.jl). Do not forget to verify that the method actually works!!!
* [Hogwild!](https://proceedings.neurips.cc/paper/2011/file/218a0aefd1d1a4be65601cc6ddc1520e-Paper.pdf)
* [Local sgd with periodic averaging: Tighter analysis and adaptive synchronization](https://proceedings.neurips.cc/paper/2019/file/c17028c9b6e0c5deaad29665d582284a-Paper.pdf)
* [Distributed optimization for deep learning with gossip exchange](https://arxiv.org/abs/1804.01852)

## Solve issues in existing projects:

### Address issues in markov decision processes (Mentor: Jan Mrkos)

Fix type stability issue in [MCTS.jl](https://github.com/JuliaPOMDP/MCTS.jl), prepare benchmarks,
and evaluate the impact of the changes. Details can be found in [this
issue](https://github.com/JuliaPOMDP/MCTS.jl/issues/59). This project will require learnind a little
bit about Markov Decision Processes if you don't know them already.

If it sounds interesting, get in touch with lecturer/lab assistant, who will connect you with Jan Mrkos.

### Extend HMil library with Retentative networks (Mentor: Tomas Pevny)

[Retentative networks](https://arxiv.org/abs/2307.08621) were recently proposed as a low-cost  alternative to Transformer models without sacrificing performance (according to authors). By implementing Retentative Networks, te HMil library will be able to learn sequences (not just sets), which might nicely extend its applicability.

### Address issues in HMil/JsonGrinder library (Mentor: Simon Mandlik)

These are open source toolboxes that are used internally in Avast. Lots of general functionality is done, but some love is needed in polishing.

- refactor the codebase using package extensions (e.g. for FillArrays)
- improve compilation time (tracking down bottlenecks with SnoopCompile and using precompile directives from PrecompileTools.jl)

Or study new metric learning approach on application in animation description

- apply machine learning on slides within presentation provide by PowToon

If it sounds interesting, get in touch with lecturer/lab assistant, who will connect you with Simon Mandlik.

## [Former projects for your inspiration](@id former_projects)

The following is a list of great projects of past years.

- [NeuralCollaborativeFiltering.jl](https://github.com/poludmik/NeuralCollaborativeFiltering.jl)
- [OptimalTrainControl.jl](https://github.com/vtfanta/OptimalTrainControl_v2.jl)
- [Urban Traffic Control](https://github.com/Matyxus/UTC_jl)
- [Directed Evolution in Silico](https://github.com/soldamatlab/DESilico.jl)
- [ImageInspector.jl](https://github.com/JuliaTeachingCTU/ImageInspector.jl) (used in our bachelor course)