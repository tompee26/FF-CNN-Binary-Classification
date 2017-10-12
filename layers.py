import tensorflow as tf


def conv_layer(batch_input, input_channel, output_channel, name="conv"):
    with tf.name_scope(name):
        weight = tf.Variable(tf.truncated_normal([4, 4, input_channel, output_channel], stddev=0.1), name="W")
        bias = tf.Variable(tf.constant(0.1, shape=[output_channel]), name="B")
        conv = tf.nn.conv2d(batch_input, weight, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + bias)
        tf.summary.histogram("weights", weight)
        tf.summary.histogram("bias", bias)
        tf.summary.histogram("activations", act)
        return act, conv


def pool_layer(batch_input, name="pool"):
    with tf.name_scope(name):
        return tf.nn.max_pool(batch_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fully_connected_layer(flattened_input, input_channel, output_channel, name="fc"):
    with tf.name_scope(name):
        weight = tf.Variable(tf.truncated_normal([input_channel, output_channel], stddev=0.1), name="W")
        bias = tf.Variable(tf.constant(0.1, shape=[output_channel]), name="B")
        act = tf.matmul(flattened_input, weight) + bias
        tf.summary.histogram("weights", weight)
        tf.summary.histogram("bias", bias)
        tf.summary.histogram("activations", act)
        return act
