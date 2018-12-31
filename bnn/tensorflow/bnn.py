import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import numpy.random as npr
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

tfd = tfp.distributions

stv_train = np.load('stv_h_train.npy')
stv_test = np.load('stv_h_test.npy')
X, y = stv_train[:, 1:], stv_train[:, 0]
y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std
Xtest, ytest = stv_test[:, 1:], stv_test[:, 0]
ytest = (ytest - y_mean) / y_std

tf.reset_default_graph()

pred_samples = 50
batch_size = 250
# X, y = build_toy_dataset(200, noise_std=0)
# (features, target, handle, training_iterator,
#  heldout_iterator) = build_input_pipeline(X, y, batch_size, np.floor(0.25 * len(X)))
(features, target, handle, training_iterator,
 heldout_iterator) = build_input_val_pipeline(X, y, Xtest, ytest, batch_size,
                                              len(Xtest))

sample_d = lambda d: tf.reduce_mean(d.sample(pred_samples), 0)

# model = tf.keras.Sequential([
#     tf.keras.layers.Reshape([8, 15, 10]),
#     tfp.layers.Convolution2DFlipout(
#         120,
#         kernel_size=2,
#         padding='SAME',
#         activation=tf.nn.relu,
#         kernel_posterior_tensor_fn=sample_d,
#         bias_posterior_tensor_fn=sample_d),
#     tf.keras.layers.BatchNormalization(),
#     # tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=[2,2], padding='SAME'),
#     tfp.layers.Convolution2DFlipout(
#         60,
#         kernel_size=3,
#         padding='SAME',
#         activation=tf.nn.relu,
#         kernel_posterior_tensor_fn=sample_d,
#         bias_posterior_tensor_fn=sample_d),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=[2,2], padding='SAME'),
#     tf.keras.layers.Flatten(),
#     tfp.layers.DenseFlipout(
#         10,
#         activation=tf.nn.relu,
#         kernel_posterior_tensor_fn=sample_d,
#         bias_posterior_tensor_fn=sample_d),
#     tf.keras.layers.BatchNormalization(),
#     tfp.layers.DenseFlipout(
#         10,
#         activation=tf.nn.relu,
#         kernel_posterior_tensor_fn=sample_d,
#         bias_posterior_tensor_fn=sample_d),
#     tf.keras.layers.BatchNormalization(),
#     tfp.layers.DenseFlipout(
#         10,
#         activation=tf.nn.relu,
#         kernel_posterior_tensor_fn=sample_d,
#         bias_posterior_tensor_fn=sample_d),
#     tf.keras.layers.BatchNormalization(),
#     tfp.layers.DenseFlipout(
#         10,
#         activation=tf.nn.relu,
#         kernel_posterior_tensor_fn=sample_d,
#         bias_posterior_tensor_fn=sample_d),
#     tf.keras.layers.BatchNormalization(),
#     tfp.layers.DenseFlipout(
#         1,
#         kernel_posterior_tensor_fn=sample_d,
#         bias_posterior_tensor_fn=sample_d),
# ])

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

# noise = tf.Variable(tf.fill([batch_size, 1], 1e-4))

preds = tf.reshape(model(tf.cast(features, tf.float32)), tf.shape(target))
# target_distribution = tfd.Normal(loc=preds, scale=noise)
# neg_log_likelihood = tf.reduce_mean(
#     -target_distribution.log_prob(tf.cast(target, tf.float32)))
neg_log_likelihood = tf.reduce_mean(
    tf.losses.mean_squared_error(tf.cast(target, tf.float32), preds))
kl = sum(model.losses) / len(X)
loss = neg_log_likelihood + kl
train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)

    train_handle = sess.run(training_iterator.string_handle())
    heldout_handle = sess.run(heldout_iterator.string_handle())
    for epoch in range(3000):
        _, ytrain, preds_train, loss_train = sess.run(
            [train_op, target, preds, loss], feed_dict={handle: train_handle})
        if not epoch % 30:
            [loss_val, xval, yval, mu_val] = sess.run(
                [neg_log_likelihood, features, target, preds],
                feed_dict={handle: heldout_handle})
            print('{} training mae {} | validation mae {}'.format(
                epoch,
                np.mean(np.abs(ytrain * y_std - preds_train * y_std)),
                np.mean(np.abs(yval * y_std - mu_val * y_std))))
