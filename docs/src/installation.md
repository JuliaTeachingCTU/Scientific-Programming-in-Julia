# [Installation](@id install)

In order to participate in the course, everyone should install a recent version of Julia together
with some text editor of choice. Furthermore during the course we will introduce some best practices
of creating/testing and distributing your own Julia code, for which we will require a GitHub
account.

We recommend to install Julia via [`juliaup`](https://github.com/JuliaLang/juliaup). We are using
the latest, *stable* version of Julia (which at the time of this writing is `v1.9`). Once you have
installed `juliaup` you can get any Julia version you want via:

```bash
juliaup add $JULIA_VERSION

# or more concretely:
juliaup add 1.9

# but please, just use the latest, stable version
```

Now you should be able to start Julia an be greated with the following:
```bash
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.9.2 (2023-07-05)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
```


## Julia IDE

There is no one way to install/develop and run Julia, which may be strange users coming from MATLAB,
but for users of general purpose languages such as Python, C++ this is quite common. Most of the
Julia programmers to date are using

- [Visual Studio Code](https://code.visualstudio.com/),
- and the corresponding [Julia extension](https://www.julia-vscode.org/).

This setup is described in a comprehensive [step-by-step
guide](https://juliateachingctu.github.io/Julia-for-Optimization-and-Learning/stable/installation/vscode/)
in our bachelor course [*Julia for Optimization &
Learning*](https://juliateachingctu.github.io/Julia-for-Optimization-and-Learning/stable/).

Note that this setup is not a strict requirement for the lectures/labs and any other text editor
with the option to send code to the terminal such as Vim (+Tmux), Emacs, or Sublime Text will
suffice.

## GitHub registration & Git setup

As one of the goals of the course is writing code that can be distributed to others, we recommend a
GitHub account, which you can create [here](https://github.com/) (unless you already have one). In
order to interact with GitHub repositories, we will be using `git`. For installation
instruction (Windows only) see the section in the bachelor
[course](https://juliateachingctu.github.io/Julia-for-Optimization-and-Learning/dev/installation/git/).
