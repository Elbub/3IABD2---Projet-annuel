import time

from numba import jit
from numba.experimental import jitclass


@jitclass
class A:
    def __init__(self, x):
        self.x = x

    def sum(self, y):
        return self.x + y


@jit(nopython=True)
def my_heavy_function(x: int, nb_iter: int):
    for i in range(nb_iter):
        for j in range(nb_iter):
            x = (x + 1.00000001 * x) / 2
    return x


@jit(nopython=True)
def my_heavy_function_wont_work(x: int, nb_iter: int):
    a = A(x)
    for i in range(nb_iter):
        for j in range(nb_iter):
            a.x = a.sum(1.00000001 * x) / 2
    return x


if __name__ == "__main__":
    prev_time = time.time()
    my_heavy_function(1, 10000)
    # print(my_heavy_function_wont_work(1, 10000)) # This will not work
    print("Elapsed time: ", time.time() - prev_time)

    prev_time = time.time()
    my_heavy_function(1, 10000)
    # print(my_heavy_function_wont_work(1, 10000)) # This will not work
    print("Elapsed time: ", time.time() - prev_time)
