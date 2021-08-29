## from introduction

Recursive call
```
fsum(x) = x
fsum(x,p...) = x+fsum(p[1],p[2:end]...)
```

Is a template how to behave:
- julia is a typed language: x, and p are of type Any
- what happens when calling:
```
fsum(1,2,3)
fsum(1,'c')
fsum(1,"c")
```

Show what "+" function do, 
```
methods(+)
```

what is multiple dispatch, how differnt it is from operator overloading.
(Stefans example from lecture)

## Show error message:
- explain why ```fsum(1,"c")``` failed.

## Introduce type hierarchy

![](from_Vasek_macha)

- Define ```fsum(x::Number,p...) = x+fsum(p[1],p[2:end]...)```
- Show how to extend to Arrays
- Define ```fsum(x::AbstractArray,p...) = x+fsum(p[1],p[2:end]...)```

- Union







