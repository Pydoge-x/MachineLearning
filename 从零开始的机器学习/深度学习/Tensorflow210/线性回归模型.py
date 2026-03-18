import tensorflow as tf

class LinearRegression(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        self.W = self.add_weight([input_dim, output_dim])
        self.bias = self.add_weight([1, output_dim])

    def call(self, inputs):

        z = tf.matmul(inputs, self.W) + self.bias

        output = tf.math.sigmoid(z)
        return output