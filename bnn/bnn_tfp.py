import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import numpy.random as npr
# import matplotlib
# matplotlib.use("Agg")
# from matplotlib import figure  # pylint: disable=g-import-not-at-top
# from matplotlib.backends import backend_agg
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

tfd = tfp.distributions


def plot_heldout_prediction(input_val,
                            y_val,
                            mu_val,
                            sigma_val,
                            fname=None,
                            n=1,
                            title=""):
    """Save a PNG plot visualizing posterior uncertainty on heldout data.
  Args:
    input_val: input locations of heldout data.
    y_val: heldout target.
    mu_val: predictive mean.
    sigma_val: predictive standard deviation.
    fname: Python `str` filename to save the plot to, or None to show.
    title: Python `str` title for the plot.
  """
    fig = figure.Figure(figsize=(9, 3 * n))
    canvas = backend_agg.FigureCanvasAgg(fig)
    for i in range(n):
        ax = fig.add_subplot(n, i + 1, 1)
        ax.plot(input_val, y_val, label='True data')
        ax.plot(input_val, mu_val, label='Predictive mean')
        lower = mu_val - sigma_val
        upper = mu_val + sigma_val
        ax.fill_between(input_val, lower, upper, label='1 std')

    fig.suptitle(title)
    fig.tight_layout()

    if fname is not None:
        canvas.print_figure(fname, format="png")
        print("saved {}".format(fname))


def build_toy_dataset(n_data=40, noise_std=0.1):
    D = 1
    rs = npr.RandomState(0)
    inputs = np.concatenate(
        [np.linspace(0, 2, num=n_data / 2),
         np.linspace(6, 8, num=n_data / 2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 4.0
    inputs = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets


def build_input_pipeline(X, y, batch_size, heldout_size):
    """Build an Iterator switching between train and heldout data."""

    # Build an iterator over training batches.
    training_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    training_batches = training_dataset.shuffle(
        len(X), reshuffle_each_iteration=True).repeat().batch(batch_size)
    training_iterator = training_batches.make_one_shot_iterator()

    # Build a iterator over the heldout set with batch_size=heldout_size,
    # i.e., return the entire heldout set as a constant.
    heldout_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    heldout_frozen = (
        heldout_dataset.take(heldout_size).repeat().batch(batch_size))
    heldout_iterator = heldout_frozen.make_one_shot_iterator()

    # Combine these into a feedable iterator that can switch between training
    # and validation inputs.
    handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    X, y = feedable_iterator.get_next()

    return X, y, handle, training_iterator, heldout_iterator


def build_input_val_pipeline(X, y, Xval, yval, batch_size, batch_heldout_size):
    """Build an Iterator switching between train and heldout data."""

    # Build an iterator over training batches.
    training_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    training_batches = training_dataset.shuffle(
        len(X), reshuffle_each_iteration=True).repeat().batch(batch_size)
    training_iterator = training_batches.make_one_shot_iterator()

    # Build a iterator over the heldout set with batch_size=heldout_size,
    # i.e., return the entire heldout set as a constant.
    heldout_dataset = tf.data.Dataset.from_tensor_slices((Xval, yval))
    heldout_frozen = (heldout_dataset.repeat().batch(batch_heldout_size))
    heldout_iterator = heldout_frozen.make_one_shot_iterator()

    # Combine these into a feedable iterator that can switch between training
    # and validation inputs.
    handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    X, y = feedable_iterator.get_next()

    return X, y, handle, training_iterator, heldout_iterator


stv_train = np.load('stv_h_train.npy')
stv_test = np.load('stv_h_test.npy')
X, y = stv_train[:, 1:], stv_train[:, 0]
Xtest, ytest = stv_test[:, 1:], stv_test[:, 0]

tf.reset_default_graph()

pred_samples = 20
batch_size = 50
# X, y = build_toy_dataset(200, noise_std=0)
# (features, target, handle, training_iterator,
#  heldout_iterator) = build_input_pipeline(X, y, batch_size, np.floor(0.25 * len(X)))
(features, target, handle,
 training_iterator, heldout_iterator) = build_input_val_pipeline(
     X, y, Xtest, ytest, batch_size, batch_size)

sample_d = lambda d: tf.reduce_mean(d.sample(pred_samples), 0)

model = tf.keras.Sequential([
    tfp.layers.DenseFlipout(
        1200,
        activation=tf.nn.relu,
        kernel_posterior_tensor_fn=sample_d,
        bias_posterior_tensor_fn=sample_d),
    tfp.layers.DenseFlipout(
        120,
        activation=tf.nn.relu,
        kernel_posterior_tensor_fn=sample_d,
        bias_posterior_tensor_fn=sample_d),
    tfp.layers.DenseFlipout(
        12,
        activation=tf.nn.relu,
        kernel_posterior_tensor_fn=sample_d,
        bias_posterior_tensor_fn=sample_d),
    tfp.layers.DenseFlipout(
        1,
        kernel_posterior_tensor_fn=sample_d,
        bias_posterior_tensor_fn=sample_d),
])

noise = tf.Variable(tf.fill([batch_size, 1], 1e-4))

preds = model(tf.cast(features, tf.float32))
target_distribution = tfd.Normal(loc=preds, scale=noise)
neg_log_likelihood = tf.reduce_mean(
    -target_distribution.log_prob(tf.cast(target, tf.float32)))
# neg_log_likelihood = tf.reduce_mean(
#     tf.pow(tf.cast(target, tf.float32) - preds, 2) / tf.pow(noise, 2))
kl = sum(model.losses) / len(X)
loss = neg_log_likelihood + kl
train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)

    train_handle = sess.run(training_iterator.string_handle())
    heldout_handle = sess.run(heldout_iterator.string_handle())
    for epoch in range(1000):
        _, ytrain, preds_train, noise_level, loss_train = sess.run(
            [train_op, target, preds, noise, loss],
            feed_dict={handle: train_handle})
        [loss_val, xval, yval, mu_val, sigma_val] = sess.run(
            [neg_log_likelihood, features, target, preds, noise],
            feed_dict={handle: heldout_handle})
        print('{} training loss {} (noise level {}) | validation loss {} '.
              format(epoch, loss_train, noise_level, loss_val))
