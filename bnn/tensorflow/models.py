import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

PRED_SAMPLES = 50

TRAINING = True


def sample_d(d):
    """Sample function for weights and biases in bayesian neural net."""
    if TRAINING:
        return tf.reduce_mean(d.sample(PRED_SAMPLES), 0)
    return d.sample()


def make_conv_net():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape([8, 15, 10]),
        tfp.layers.Convolution2DFlipout(
            120,
            kernel_size=2,
            padding='SAME',
            activation=tf.nn.relu,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=[2,2], padding='SAME'),
        tfp.layers.Convolution2DFlipout(
            60,
            kernel_size=3,
            padding='SAME',
            activation=tf.nn.relu,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2], padding='SAME'),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(
            10,
            activation=tf.nn.relu,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.DenseFlipout(
            10,
            activation=tf.nn.relu,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.DenseFlipout(
            10,
            activation=tf.nn.relu,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.DenseFlipout(
            10,
            activation=tf.nn.relu,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.DenseFlipout(
            1,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
    ])
    return model


def make_mlp_net():
    model = tf.keras.Sequential([
        tfp.layers.DenseFlipout(
            10,
            activation=tf.nn.relu,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.DenseFlipout(
            10,
            activation=tf.nn.relu,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.DenseFlipout(
            10,
            activation=tf.nn.relu,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.DenseFlipout(
            10,
            activation=tf.nn.relu,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.DenseFlipout(
            10,
            activation=tf.nn.relu,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.DenseFlipout(
            1,
            kernel_posterior_tensor_fn=sample_d,
            bias_posterior_tensor_fn=sample_d),
    ])
    return model
