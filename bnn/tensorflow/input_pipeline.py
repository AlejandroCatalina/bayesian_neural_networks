import tensorflow as tf


def build_input_pipeline(X, y, batch_size, heldout_size, graph):
    """Build an Iterator switching between train and heldout data."""

    with graph.as_default():
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
            handle, training_batches.output_types,
            training_batches.output_shapes)
        X, y = feedable_iterator.get_next()

    return X, y, handle, training_iterator, heldout_iterator, graph


def build_input_val_pipeline(X, y, Xval, yval, batch_size, batch_heldout_size,
                             graph):
    """Build an Iterator switching between train and heldout data."""

    with graph.as_default():
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
            handle, training_batches.output_types,
            training_batches.output_shapes)
        X, y = feedable_iterator.get_next()

    return X, y, handle, training_iterator, heldout_iterator, graph
