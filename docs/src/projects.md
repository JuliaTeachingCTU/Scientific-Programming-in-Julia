# [Projects](@id projects)

We want you to use Julia for something that is acutally useful for you.
Therefore you can choose your final, graded project very freely.
We will discuss your ideas with you individually and come up with a sufficiently
extensive project together.

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

## Distributed Optimization Package

One click distributed optimization is at the heart of other machine learning 
and optimization libraries such as pytorch, however some equivalents are 
missing in the Julia's Flux ecosystem. The goal of this project is to explore,
implement and compare at least two state-of-the-art methods of distributed 
gradient descent on data that will be provided for you.

## A Rule Learning Algorithm

[Rule-based models](https://christophm.github.io/interpretable-ml-book/rules.html)
are simple and very interpretable models that have been around for a long time
and are gaining popularity again.
The goal of this project is to implement a
[sequential covering](https://christophm.github.io/interpretable-ml-book/rules.html#sequential-covering)
algorithm called [`RIPPER`](http://www.cs.utsa.edu/~bylander/cs6243/cohen95ripper.pdf)
and evaluate it on a number of datasets.
