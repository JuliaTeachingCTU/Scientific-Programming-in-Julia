## Design Patterns

SOLID: Single Responsibility, Open/Closed, Liskov Substitution, Interface
Segregation, Dependency Inversion
DRY: Don't Repeat Yourself
KISS: Keep It Simple, Stupid!
POLA: Principle of Least Astonishment
YAGNI: You Aren't Gonna Need It == (overengineering)
POLP: Principle of Least Privilege == 


### Solid

Single Responsibility Principle

The Single Responsibility Principle states that every module, class, and function should be
responsible for a single functional objective. There should be only one reason to make any
changes.
The benefits of this principle are listed here:
- The programmer can focus on a single context during development.
- The size of each component is smaller.


Open/Closed Principle

The Open/Closed Principle states that every module should be *open* for extension but
*closed* for modification. It is necessary to distinguish between enhancement and
extension—enhancement refers to a core improvement of the existing module, while an
extension is considered an add-on that provides additional functionality.
The following are the benefits of this principle:
- Existing components can be easily reused to derive new functionalities.
- Components are loosely coupled so it is easier to replace them without affecting the existing functionality.


Liskov Substitution Principle

The Liskov Substitution Principle states that a program that accepts type T can also accept
type S (which is a subtype of T), without any change in behavior or intended outcome.
The following are the benefits of this principle:
- A function can be reused for any subtype passed in the arguments.


Interface Segregation Principle
The Interface Segregation Principle states that a client should not be forced to implement
interfaces that it does not need to use.
The following are the benefits of this principle:
- Software components are more modular and reusable.
- New implementations can be created more easily.


Dependency Inversion Principle
The Dependency Inversion Principle states that high-level classes should not depend on
low-level classes; instead, high-level classes should depend on an abstraction that low-level
classes implement

The following are the benefits of this principle:
- Components are more decoupled.
- The system becomes more flexible and can adapt to changes more easily. Low-level components can be replaced without affecting high-level components.


