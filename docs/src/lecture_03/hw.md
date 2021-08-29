# Homework 3

Implement a gendered sheep that can only reproduce with another sheep of
opposite gender.

The gendered sheep need an additonal field. As you cannot inherit from a
concrete type in julia, you will have to forward the `eats` and `eat!` methods.
In order to get the custom reproduction behaviour you have to overload the
`reproduce!` function.
