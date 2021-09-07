## Ordinary ODE

Consider
$$
\frac{du}{dt}+N(u)=0
$$

which may have many special cases, e.g. the 

$$
burg
$$

Consider a surrogate function $f$ to be represented by a neural network $f=NN_\theta(x,t)$.

Define two loss functions:
$$
\begin{align}
MSEu = \frac{1}{N_u}\sum_{i=1}^{N_u} | f(x_u^i,t_u^i ) - u^i |^2
\end{align}
$$

