var documenterSearchIndex = {"docs":
[{"location":"projects/#projects","page":"Projects","title":"Projects","text":"","category":"section"},{"location":"projects/","page":"Projects","title":"Projects","text":"The goal of the project should be to create something, which is actually useful. Therefore we offer a lot of freedom in how the project will look like with the condition that you should spent around 60 hours on it (this number was derived as follows: each credit is worth 30 hours minus 13 lectures + labs minus 10 homeworks 2 hours each) and you should demonstrate some skills in solving the project. In general, we can distinguish three types of project depending on the beneficiary:","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"You benefit: Use / try to solve a well known problem using Julia language,\nOur group: work with your tutors on a topic researched in the AIC group, \nJulia community: choose an issue in a registered Julia project you like and fix it (documentation issues are possible but the resulting documentation should be very nice.).","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"The project should be of sufficient complexity that verify your skill of the language (to be agreed individually).","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"Below, we list some potential projects for inspiration.","category":"page"},{"location":"projects/#Improving-documentation","page":"Projects","title":"Improving documentation","text":"","category":"section"},{"location":"projects/","page":"Projects","title":"Projects","text":"Improve documentation of one of these projects","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"SymbolicPlanners.jl\nDaggerFlux.jl for model-based parallelization of large Neural Networks (see JuliaCon 2022 for exposition of the package)\nFluxDistributed.jl for data-based parallelization of large Neural Networks (see JuliaCon 2022 for exposition of the package)","category":"page"},{"location":"projects/#Implementing-new-things","page":"Projects","title":"Implementing new things","text":"","category":"section"},{"location":"projects/#The-Equation-Learner-And-Its-Symbolic-Representation","page":"Projects","title":"The Equation Learner And Its Symbolic Representation","text":"","category":"section"},{"location":"projects/","page":"Projects","title":"Projects","text":"In many scientific and engineering one searches for interpretable (i.e. human-understandable) models instead of the black-box function approximators that neural networks provide. The equation learner (EQL) is one approach that can identify concise equations that describe a given dataset.","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"The EQL is essentially a neural network with different unary or binary activation functions at each indiviual unit. The network weights are regularized during training to obtain a sparse model which hopefully results in a model that represents a simple equation.","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"The goal of this project is to implement the EQL, and if there is enough time the improved equation learner (iEQL). The equation learners should be tested on a few toy problems (possibly inspired by the tasks in the papers).  Finally, you will implement functionality that can transform the learned model into a symbolic, human readable, and exectuable Julia expression.","category":"page"},{"location":"projects/#Learning-Large-Language-Models-with-reduced-precition","page":"Projects","title":"Learning Large Language Models with reduced precition","text":"","category":"section"},{"location":"projects/","page":"Projects","title":"Projects","text":"Large Language Models ((Chat) GPT, LLama, Falcon, Palm, ...) are huge. A recent trend is to perform optimization in reduced precision, for example in int8 instead of Float32. Such feature is currently missing in Julia ecosystem and this project should be about bringing this to the community (for an introduction, read these blogs *LLM-int8 and emergent features**, *A gentle introduction to 8-bit Matrix Multiplication). The goal would be to implement this as an additional type of Number / Matrix and overload multiplication on CPU (and ideally on GPU) to make it transparent for neural networks? What I will learn? In this project, you will learn a lot about the (simplicity of) implementation of deep learning libraries and you will practice abstraction of Julia's types. You can furthermore learn about GPU Kernel programming and Transformers.jl library.","category":"page"},{"location":"projects/#Planning-algorithms","page":"Projects","title":"Planning algorithms","text":"","category":"section"},{"location":"projects/","page":"Projects","title":"Projects","text":"Extend SymbolicPlanners.jl with the mm-ϵ variant of the bi-directional search MM: A bidirectional search algorithm that is guaranteed to meet in the middle. This pull request might be very helpful in understanding better the library.","category":"page"},{"location":"projects/#A-Rule-Learning-Algorithms","page":"Projects","title":"A Rule Learning Algorithms","text":"","category":"section"},{"location":"projects/","page":"Projects","title":"Projects","text":"Rule-based models are simple and very interpretable models that have been around for a long time and are gaining popularity again. The goal of this project is to implement one of these algorithms","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"sequential covering","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"algorithm called RIPPER and evaluate it on a number of datasets.","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"Learning Certifiably Optimal Rule Lists for Categorical Data\nBoolean decision rules via column generation\nLearning Optimal Decision Trees with SAT\nA SAT-based approach to learn explainable decision sets","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"To increase the impact of the project, consider interfacing it with MLJ.jl","category":"page"},{"location":"projects/#Parallel-optimization","page":"Projects","title":"Parallel optimization","text":"","category":"section"},{"location":"projects/","page":"Projects","title":"Projects","text":"Implement one of the following algorithms to train neural networks in parallel. Can be implemented in a separate package or consider extending FluxDistributed.jl. Do not forget to verify that the method actually works!!!","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"Hogwild!\nLocal sgd with periodic averaging: Tighter analysis and adaptive synchronization\nDistributed optimization for deep learning with gossip exchange","category":"page"},{"location":"projects/#Improving-support-for-multi-threadding-functions-in-NNLib","page":"Projects","title":"Improving support for multi-threadding functions in NNLib","text":"","category":"section"},{"location":"projects/","page":"Projects","title":"Projects","text":"NNlib.jl is a workhorse library for deep learning in Julia (it powers Flux.jl). Yet most of their functions are single-threaded. The task is to choose few of them (e.g. logitcrossentropy or application of non-linearity) and make them multi-threaded. Ideally, you should make a workable pull request that will be accepted by the community. Warning: this will require interaction with the Flux community","category":"page"},{"location":"projects/#Project-requirements","page":"Projects","title":"Project requirements","text":"","category":"section"},{"location":"projects/","page":"Projects","title":"Projects","text":"The goal of the semestral project is to create a Julia pkg with reusable, properly tested and documented code. We have given you some options of topics, as well as the freedom to choose something that could be useful for your research or other subjects. In general we are looking for something where performance may be crucial such as data processing, optimization or equation solving.","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"In practice the project should follow roughly this tree structure","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":".\n├── scripts\n│\t├── run_example.jl\t\t\t# one or more examples showing the capabilities of the pkg\n│\t├── Project.toml \t\t\t# YOUR_PROJECT should be added here with develop command with rel path\n│\t└── Manifest.toml \t\t\t# should be committed as it allows to reconstruct the environment exactly\n├── src\n│\t├── YOUR_PROJECT.jl \t\t# ideally only some top level code such as imports and exports, rest of the code included from other files\n│\t├── src1.jl \t\t\t\t# source files structured in some logical chunks\n│\t└── src2.jl\n├── test\n│\t├── runtest.jl              # contains either all the tests or just includes them from other files\n│\t├── Project.toml  \t\t\t# lists some additional test dependencies\n│\t└── Manifest.toml   \t\t# usually not committed to git as it is generated on the fly\n├── README.md \t\t\t\t\t# describes in short what the pkg does and how to install pkg (e.g. some external deps) and run the example\n├── Project.toml  \t\t\t\t# lists all the pkg dependencies\n└── Manifest.toml  \t\t\t\t# usually not committed to git as the requirements may be to restrictive","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"The first thing that we will look at is README.md, which should warn us if there are some special installation steps, that cannot be handled with Julia's Pkg system. For example if some 3rd party binary dependency with license is required. Secondly we will try to run tests in the test folder, which should run and not fail and should cover at least some functionality of the pkg. Thirdly and most importantly we will instantiate environment in scripts and test if the example runs correctly. Lastly we will focus on documentation in terms of code readability, docstrings and inline comments. ","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"Only after all this we may look at the extent of the project and it's difficulty, which may help us in deciding between grades. ","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"Nice to have things, which are not strictly required but obviously improves the score.","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"Ideally the project should be hosted on GitHub, which could have the continuous integration/testing set up.\nInclude some benchmark and profiling code in your examples, which can show us how well you have dealt with the question of performance.\nSome parallelization attempts either by multi-processing, multi-threadding, or CUDA. Do not forget to show the improvement.\nDocumentation with a webpage using Documenter.jl.","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"Here are some examples of how the project could look like:","category":"page"},{"location":"projects/","page":"Projects","title":"Projects","text":"ImageInspector","category":"page"},{"location":"installation/#install","page":"Installation","title":"Installation","text":"","category":"section"},{"location":"installation/","page":"Installation","title":"Installation","text":"In order to participate in the course, everyone should install a recent version of Julia together with some text editor of choice. Furthermore during the course we will introduce some best practices of creating/testing and distributing your own Julia code, for which we will require a GitHub account.","category":"page"},{"location":"installation/#Julia-IDE","page":"Installation","title":"Julia IDE","text":"","category":"section"},{"location":"installation/","page":"Installation","title":"Installation","text":"There is no one way to install/develop and run Julia, which may be strange users coming from MATLAB, but for users of general purpose languages such as Python, C++ this is quite common. As of 2020 the most widely adopted way is in combination with the VSCode editor, for which there is an officially supported Julia extension. Moreover this setup is the same as with our bachelor course, which has provided an extensive tutorial mainly in case of installation on Windows machines, here. If you are using any other supported platform, you can use the guide as well replacing some steps with your system specifics (having the julia executable in path or as an alias is a plus). When deciding which version to download we recommend the latest stable release as of September 2022, 1.8.x. ","category":"page"},{"location":"installation/","page":"Installation","title":"Installation","text":"Note that this setup is not a strict requirement for the lectures/labs and any other text editor with the option to send code to the terminal, such as Sublime Text, Vim+tmux or Atom will suffice (a major convenience when dealing with programming languages that support interactivity through a Read-Eval-Print Loop - REPL).","category":"page"},{"location":"installation/#GitHub-registration-and-Git-setup","page":"Installation","title":"GitHub registration & Git setup","text":"","category":"section"},{"location":"installation/","page":"Installation","title":"Installation","text":"As one of the goals of the course is writing code that can be distributed to others, we require a GitHub account, which you can create here (unless you already have one). In order to interact with GitHub repositories, we will be using git client. For installation instruction (Windows only) see the section in the bachelor course.","category":"page"},{"location":"","page":"Home","title":"Home","text":"<img class=\"docs-light-only\"; src=\"https://raw.githubusercontent.com/JuliaTeachingCTU/JuliaCTUGraphics/master/logo/Scientific-Programming-in-Julia-logo.svg\"; alt=\"Scientific Programming in Julia logo\"; max-width: 100%; height: auto>\n<img class=\"docs-dark-only\"; src=\"https://raw.githubusercontent.com/JuliaTeachingCTU/JuliaCTUGraphics/master/logo/Scientific-Programming-in-Julia-logo-dark.svg\"; alt=\"Scientific Programming in Julia logo\"; max-width: 100%; height: auto;>","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Plots\nENV[\"GKSwstype\"] = \"100\"\ngr()","category":"page"},{"location":"","page":"Home","title":"Home","text":"Scientific Programming requires the highest performance but we also want to write very high level code to enable rapid prototyping and avoid error prone, low level implementations.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The Julia programming language is designed with exactly those requirements of scientific computing in mind.  In this course we will show you how to make use of the tools and advantages that jit-compiled Julia provides over dynamic, high-level languages like Python or lower level languages like C++.","category":"page"},{"location":"","page":"Home","title":"Home","text":"<figure>\n  <img src=\"assets/dual.svg\" alt=\"my alt text\"/>\n  <figcaption>\n    Learn the power of abstraction.\n    Example: The essence of <a href=\"https://juliadiff.org/ForwardDiff.jl/dev/dev/how_it_works/\">forward mode</a> automatic differentiation.\n  </figcaption>\n</figure>","category":"page"},{"location":"","page":"Home","title":"Home","text":"Before joining the course, consider reading the following two blog posts to figure out if Julia is a language in which you want to invest your time.","category":"page"},{"location":"","page":"Home","title":"Home","text":"What is great about Julia.\nWhat is bad about Julia.","category":"page"},{"location":"#What-will-you-learn?","page":"Home","title":"What will you learn?","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"First and foremost you will learn how to think julia - meaning how write fast, extensible, reusable, and easy-to-read code using things like optional typing, multiple dispatch, and functional programming concepts.  The later part of the course will teach you how to use more advanced concepts like language introspection, metaprogramming, and symbolic computing. Amonst others you will implement your own automatic differetiation (the backbone of modern machine learning) package based on these advanced techniques that can transform intermediate representations of Julia code.","category":"page"},{"location":"#Organization","page":"Home","title":"Organization","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This course webpage contains all information about the course that you need, including lecture notes, lab instructions, and homeworks. The official format of the course is 2+2 (2h lectures/2h labs per week) for 4 credits.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The official course code is: B0M36SPJ and the timetable for the winter semester 2022 can be found here.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The course will be graded based on points from your homework (max. 20 points) and points from a final project (max. 30 points).","category":"page"},{"location":"","page":"Home","title":"Home","text":"Below is a table that shows which lectures have homeworks (and their points).","category":"page"},{"location":"","page":"Home","title":"Home","text":"Homework 1 2 3 4 5 6 7 8 9 10 11 12 13\nPoints 2 2 2 2 2 2 2 2 - 2 - 2 -","category":"page"},{"location":"","page":"Home","title":"Home","text":"Hint: The first few homeworks are easier. Use them to fill up your points.","category":"page"},{"location":"#final_project","page":"Home","title":"Final project","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The final project will be individually agreed on for each student. Ideally you can use this project to solve a problem you have e.g. in your thesis, but don't worry - if you cannot come up with an own project idea, we will suggest one to you. More info and project suggestion can be found here.","category":"page"},{"location":"#Grading","page":"Home","title":"Grading","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Your points from the homeworks and the final project are summed and graded by the standard grading scale below.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Grade A B C D E F\nPoints 45-50 40-44 35-39 30-34 25-29 0-25","category":"page"},{"location":"#emails","page":"Home","title":"Teachers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"– E-mail Room Role\nTomáš Pevný pevnak@protonmail.ch KN:E-406 Lecturer\nVašek Šmídl smidlva1@fjfi.cvut.cz KN:E-333 Lecturer\nMatěj Zorek zorekmat@fel.cvut.cz KN:E-333 Lab Instructor\nNiklas Heim heimnikl@fel.cvut.cz KN:E-333 Lab Instructor","category":"page"},{"location":"#Prerequisites","page":"Home","title":"Prerequisites","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"There are no hard requirements to take the course, but if you are not at all familiar with Julia we recommend you to take Julia for Optimization and Learning before enrolling in this course. The Functional Programming course also contains some helpful concepts for this course. And knowledge about computer hardware, namely basics of how CPU works, how it interacts with memory through caches, and basics of multi-threadding certainly helps.","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Official documentation\nWorkflow tips, and what is new in v1.9\nThink Julia: How to Think Like a Computer Scientist\nFrom Zero to Julia!\nWikiBooks\nJustin Krumbiel's excellent introduction to the package manager.\njuliadatascience.io contains an excellent introduction to plotting with Makie.\nMIT Course: Julia Computation\nTim Holy's Advanced Scientific Computing","category":"page"},{"location":"how_to_submit_hw/#homeworks","page":"Homework submission","title":"Homework submission","text":"","category":"section"},{"location":"how_to_submit_hw/","page":"Homework submission","title":"Homework submission","text":"This document should describe the homework submission procedure.","category":"page"}]
}
