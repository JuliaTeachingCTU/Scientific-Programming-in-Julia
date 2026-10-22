# from timeit import timeit
import time
import numpy as np
from numpy.linalg import matrix_power as mpow
import jax
from jax import random
from jax import jit
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

key = random.PRNGKey(758493)

def timeit(f, *args, number=100):
    # compile
    f(*args)
    times = []
    for _ in range(number):
        t1 = time.time()
        f(*args)
        t2 = time.time()
        times.append(t2-t1)
    return np.min(times), np.std(times)


def run_and_save(f, argss: list[tuple], filename: str):
    ms = []
    for args in argss:
        (m,s) = timeit(f, *args)
        ms.append(m)
    np.savetxt(filename, np.array(ms)*10**6)


@jit
def f(x):
    return 3*x**3 + 2*x**2 + x + 1

def g(x):
    return 3*x**3 + 2*x**2 + x + 1


run_and_save(g, [
    (np.random.rand(10),),
    (np.random.rand(100),),
    (np.random.rand(1000),),
    (np.random.rand(10000),),
    (np.random.rand(100000),),
    (np.random.rand(1000000),),
    (np.random.rand(10000000),),
], "numpy.txt")

run_and_save(f, [
    (random.uniform(key, shape=(10,), dtype=jnp.float64),),
    (random.uniform(key, shape=(100,), dtype=jnp.float64),),
    (random.uniform(key, shape=(1000,), dtype=jnp.float64),),
    (random.uniform(key, shape=(10000,), dtype=jnp.float64),),
    (random.uniform(key, shape=(100000,), dtype=jnp.float64),),
    (random.uniform(key, shape=(1000000,), dtype=jnp.float64),),
    (random.uniform(key, shape=(10000000,), dtype=jnp.float64),),
], "jax.txt")

# n = 1000
# xnp = np.random.rand(n)
# xjx = random.uniform(key, shape=(n,))
# 
# f(xjx)
# 
# a = 10**6
# m, s = timeit(g, xnp)
# print(f"Numpy {m*a:.3f} μs")
# 
# m, s = timeit(f, xjx)
# print(f"JAX   {m*a:.3f} μs")

