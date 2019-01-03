import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import numpy.random as npr
import os

# local imports
import models
from data import build_toy_dataset
from models import make_mlp_net, make_conv_net
from input_pipeline import build_input_pipeline, build_input_val_pipeline

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

tfd = tfp.distributions

stv_train = np.load('../stv_h_train.npy')
stv_test = np.load('../stv_h_test.npy')
X, y = stv_train[:, 1:], stv_train[:, 0]

# scale y
y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std

Xtest, ytest = stv_test[:, 1:], stv_test[:, 0]
ytest = (ytest - y_mean) / y_std

tf.reset_default_graph()

batch_size = 250
# X, y = build_toy_dataset(200, noise_std=0)
# (features, target, handle, training_iterator,
#  heldout_iterator) = build_input_pipeline(X, y, batch_size, np.floor(0.25 * len(X)))
(features, target, handle, training_iterator,
 heldout_iterator) = build_input_val_pipeline(X, y, Xtest, ytest, batch_size,
                                              len(Xtest))

model = make_mlp_net()

preds = tf.reshape(model(tf.cast(features, tf.float32)), tf.shape(target))
neg_log_likelihood = tf.reduce_mean(
    tf.losses.mean_squared_error(tf.cast(target, tf.float32), preds))
kl = sum(model.losses) / len(X)
loss = neg_log_likelihood + kl
train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# predict operation
predict_op = tf.stack([
    tf.reshape(model(tf.cast(features, tf.float32)), tf.shape(target))
    for _ in range(models.PRED_SAMPLES)
])

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
            models.TRAINING = False
            [loss_val, xval, yval, preds_val] = sess.run(
                [neg_log_likelihood, features, target, predict_op],
                feed_dict={handle: heldout_handle})
            preds_val = np.mean(preds_val, axis=0)
            print('{} training mae {} | validation mae {}'.format(
                epoch,
                np.mean(np.abs(ytrain * y_std - preds_train * y_std)),
                np.mean(np.abs(yval * y_std - preds_val * y_std))))
            models.TRAINING = True
