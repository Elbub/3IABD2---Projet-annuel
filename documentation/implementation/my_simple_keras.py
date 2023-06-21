import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange


class MyModel:
    def __call__(self, batch_X):
        raise NotImplementedError

    def get_trainable_variables(self):
        raise NotImplementedError

    def my_predict(self, batch_X):
        if len(batch_X.shape) == 1:
            batch_X = np.expand_dims(batch_X, axis=1)
        return self(batch_X)

    def my_fit(self, dataset_X, dataset_Y, epochs: int, batch_size: int, learning_rate: float):

        t = trange(epochs, desc="loss :", leave=True)

        for e in t:
            sample_indices = np.arange(0, len(dataset_X))
            np.random.shuffle(sample_indices)
            dataset_X = tf.gather(dataset_X, sample_indices)
            dataset_Y = tf.gather(dataset_Y, sample_indices)

            mean_epoch_loss = 0.0
            mean_steps = 0
            for batch_id in range(len(dataset_X) // batch_size + 1):
                batch_X = dataset_X[batch_id * batch_size: (batch_id + 1) * batch_size]
                batch_Y = dataset_Y[batch_id * batch_size: (batch_id + 1) * batch_size]
                if len(batch_X) == 0:
                    continue
                batch_loss = self.my_train_step(batch_X, batch_Y, learning_rate)
                mean_epoch_loss = ((mean_epoch_loss * mean_steps) + batch_loss) / (mean_steps + 1)
            t.set_description(f'loss : {mean_epoch_loss}')
            t.refresh()

    def my_train_step(self, batch_X, batch_Y, learning_rate: float) -> float:
        if len(batch_Y.shape) == 1:
            batch_Y = np.expand_dims(batch_Y, axis=1)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(self.my_predict(batch_X) - batch_Y))
        trainable_variables = self.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        for (grad, v) in zip(gradients, trainable_variables):
            v.assign_sub(learning_rate * grad)
        return loss


class MyLinearModel(MyModel):
    def __init__(self, input_size: int, output_size: int):
        self.w = tf.Variable(tf.random.uniform((input_size, output_size)))
        self.b = tf.Variable(tf.zeros((output_size,)))

    def __call__(self, batch_X):
        return tf.matmul(batch_X, self.w) + self.b

    def get_trainable_variables(self):
        return [self.w, self.b]


class MyMLPModel(MyModel):
    def __init__(self, npl: [int]): # npl = number of neurons per layer, ex: [1, 16, 16, 1]
        self.layers_params = []
        self.num_layers = len(npl) - 1
        for l in range(len(npl) - 1):
            w = tf.Variable(tf.random.uniform((npl[l], npl[l + 1]), minval=-1.0, maxval=1.0))
            b = tf.Variable(tf.zeros((npl[l + 1],)))
            self.layers_params.append((w, b))

    def __call__(self, batch_X):
        output = batch_X
        for (l, (w, b)) in enumerate(self.layers_params):
            output = tf.matmul(output, w) + b
            if l < self.num_layers - 1:
                output = tf.tanh(output)
        return output

    def get_trainable_variables(self):
        return [w for (w, _) in self.layers_params] + [b for (_, b) in self.layers_params]



if __name__ == "__main__":
    points_X = []
    points_Y = []
    for i in range(100):
        points_X.append(i / 100.0)
        # points_Y.append(3 * i / 100.0 + 4 + np.random.normal(0, 0.05))
        points_Y.append(3 * np.sin(i / 100.0 * 5 + 4) + np.random.normal(0, 0.05))

    points_X = tf.constant(points_X, dtype=tf.float32)
    points_Y = tf.constant(points_Y, dtype=tf.float32)
    # model = MyLinearModel(1, 1)
    model = MyMLPModel([1, 16, 16, 1])


    predicted_points_Y = model.my_predict(points_X)

    import matplotlib.pyplot as plt

    plt.scatter(points_X, points_Y, c='b')
    plt.scatter(points_X, predicted_points_Y, c='r')
    plt.show()


    model.my_fit(points_X, points_Y, 5000, 50, 0.01)
    predicted_points_Y = model.my_predict(points_X)
    print(model.get_trainable_variables())

    import matplotlib.pyplot as plt

    plt.scatter(points_X, points_Y, c='b')
    plt.scatter(points_X, predicted_points_Y, c='r')
    plt.show()
