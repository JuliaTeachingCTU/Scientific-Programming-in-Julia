# [Installation](@id install)
In order to participate in the course, everyone should install a recent version of Julia together with some text editor of choice. Furthermore during the course we will introduce some best practices of creating/testing and distributing your own Julia code, for which we will require a GitHub account.

## Julia IDE
As there does not exist any proper IDE for Julia yet, there is no one way to install/develop and run Julia. However following the bachelor course *TODO LINK* we will be using currently the most widely adopted way to code in Julia and that is in combination with VSCode editor. If you are still rocking Windows machine there is a great guide on getting such setup working on the bachelor course. *TODO LINK or COPY* When deciding which version to download we recommend the latest stable release as of August 2021, `1.6.x`, which has some quality of life improvements mainly with regards to exception readability. This is however not a strict requirement and any text editor with the option to send code to the terminal, such as Sublime Text, Vim+tmux or Atom will suffice (a major convenience when dealing with programming languages that support interactivity through a Read-Eval-Print Loop - REPL).

## GitHub registration & Git setup
If you are familiar with git and GitHub, we recommend following the guide at *TODO LINK or COPY* to create your own GitHub account and install a git client.


## Advanced setup (not required)
More advanced users can experiment with the following setups on Windows and/or Linux/MacOS platforms. More specifically we believe that knowing how to run any language inside a Docker container helps to understand the problem of dependencies and reproducible code in more depth than with a simple local installation. Moreover given the convenience of remote development in VSCode both of the setups feel almost like a local installation.

### Julia in WSL
This setup is specific to Windows 10, which as of 2016 allows to run any Linux distribution almost natively on top of the Windows kernel.

### Julia in Docker
- Install Docker client
- Setup the permissions
- Build a simple docker with Julia Hello world
- Caveats around such setup - GUI, dependencies, file system, running as root







