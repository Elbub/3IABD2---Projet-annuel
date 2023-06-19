import tensorflow as tf
from keras.utils import plot_model


if __name__ == "__main__":
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, input_shape=(20,)))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Dense(1))
    print(model.predict(tf.zeros( (3, 20) )))
    model.save('my_model.keras')
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    model.summary()

