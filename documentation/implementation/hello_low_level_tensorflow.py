import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def my_main():
    mat = np.zeros((3, 3))
    print(mat)
    mat = np.identity(3)
    print(mat)
    mat = np.ones((3, 3))
    print(mat)
    mat = np.random.random((3, 3))
    print(mat)
    print(np.dot(mat, mat))
    print(np.matmul(mat, mat))
    invmat = np.linalg.inv(mat)
    print(invmat)
    print(np.matmul(mat, invmat))
    print(mat[:, 1])
    print(mat[mat > 0.5])

    mat = tf.zeros((3, 3))
    print(mat)
    mat = tf.eye(3)
    print(mat)
    mat = tf.ones((3, 3))
    print(mat)
    mat = tf.random.uniform((3, 3))
    print(mat)
    # print(tf.dot(mat, mat)) not working
    print(tf.matmul(mat, mat))
    invmat = tf.linalg.inv(mat)
    print(invmat)
    print(tf.matmul(mat, invmat))

    x = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        y = 42.0 * x
    print(tape.gradient(y, x))

    points_X = []
    points_Y = []
    for i in range(100):
        points_X.append(i / 100.0)
        points_Y.append(2 * np.sin(i / 100.0 * 3 + 4) + np.random.normal(0, 0.05))
    print(points_X)
    print(points_Y)

    points_X = tf.constant(points_X, dtype=tf.float32)
    points_Y = tf.constant(points_Y, dtype=tf.float32)

    a = tf.Variable(-0.2)
    b = tf.Variable(-0.1)
    c = tf.Variable(0.1)
    start_time = time.time()
    for i in range(10000):
        train_step(a, b, c, points_X, points_Y)
    print("Elapsed time: ", time.time() - start_time)

    predicted_points_Y = a * points_X ** 2 + c * points_X + b
    plt.scatter(points_X, points_Y, c='b')
    plt.scatter(points_X, predicted_points_Y, c='r')
    plt.show()

@tf.function
def train_step(a, b, c, points_X, points_Y):
    with tf.GradientTape() as tape:
        y = a * points_X ** 2 + c * points_X + b
        loss = tf.reduce_mean(tf.square(y - points_Y))
    (grad_a, grad_b, grad_c) = tape.gradient(loss, (a, b, c))
    print(grad_a, grad_b, grad_c)
    print(loss)
    a.assign_sub(0.01 * grad_a)
    b.assign_sub(0.01 * grad_b)
    c.assign_sub(0.01 * grad_c)


if __name__ == "__main__":
    my_main()
