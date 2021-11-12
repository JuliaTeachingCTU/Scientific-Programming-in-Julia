# Homework 08

In this homework you will write an additional rule to compute the derivative
of the Babylonian square root

```@example hw08
babysqrt(x, t=(1+x)/2, n=10) = n==0 ? t : babysqrt(x, (t+x/t)/2, n-1)
```

And of the function
```@example hw08
f(x) = sin(babysqrt(x))
```
