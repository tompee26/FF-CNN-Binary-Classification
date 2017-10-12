import tensorflow as tf
import layers as ly
import datareader as dr
import utils

""" Input image dimension (square) """
INPUT_IMAGE_DIMENSION = 28

""" Input image channel (greyscale) """
INPUT_IMAGE_CHANNELS = 1

""" Training step size """
STEP_SIZE = 800

""" Input batch size"""
BATCH_SIZE = 30

""" Fully connected layer neuron count """
EXTRA_FC_LAYER_NODES = 1024

""" Output one hot vector size """
OUTPUT_VECTOR_SIZE = 2

""" Test step size """
TEST_STEP_SIZE = 500


def conv_net_model(learning_rate, graph_dir, train_dir, test_dir, extra_fc_layer=False):
    """
    The feed forward convolutional neural network model

    Hyper parameters include learning rate, number of convolutional layers and
    fully connected layers. (Currently TBD)

    """
    # Reset graphs
    tf.reset_default_graph()

    # Create a tensorflow session
    sess = tf.Session()

    # Create placeholders
    x = tf.placeholder(dtype=tf.float32,
                       shape=[None, INPUT_IMAGE_DIMENSION, INPUT_IMAGE_DIMENSION, INPUT_IMAGE_CHANNELS],
                       name="x")
    y = tf.placeholder(dtype=tf.float32,
                       shape=[None, OUTPUT_VECTOR_SIZE],
                       name="y")
    # Visualize input x
    tf.summary.image("input", x, BATCH_SIZE)

    # First convolutional layer
    conv1, v = ly.conv_layer(x, INPUT_IMAGE_CHANNELS, 32, name="conv1")

    # Visualize convolution output
    utils.visualize_conv(v, INPUT_IMAGE_DIMENSION, 32, "raw_conv1")
    # Visualize relu activated convolution output
    utils.visualize_conv(conv1, INPUT_IMAGE_DIMENSION, 32, "relu_conv1")

    # First pooling
    pool1 = ly.pool_layer(conv1, name="pool1")
    image_dimension = int(INPUT_IMAGE_DIMENSION / 2)
    # Visualize first pooling
    utils.visualize_conv(pool1, image_dimension, 32, "pool1")

    # Flatten input
    flattened = tf.reshape(pool1, shape=[-1, image_dimension * image_dimension * 32])

    # Create fully connected layer
    if extra_fc_layer:
        fc_layer = ly.fully_connected_layer(flattened,
                                            image_dimension * image_dimension * 32,
                                            EXTRA_FC_LAYER_NODES,
                                            name="fc_layer")
        logits = ly.fully_connected_layer(fc_layer,
                                          EXTRA_FC_LAYER_NODES,
                                          OUTPUT_VECTOR_SIZE,
                                          name="logits")
    else:
        logits = ly.fully_connected_layer(flattened,
                                          image_dimension * image_dimension * 32,
                                          OUTPUT_VECTOR_SIZE,
                                          name="logits")

    # Create loss function
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar("cross_entropy", cross_entropy)

    # Create optimizer
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # Compute accuracy
    with tf.name_scope("accuracy"):
        # argmax gets the highest value in a given dimension (in this case, dimension 1)
        # equal checks if the label is equal to the computed logits
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        # tf.reduce_mean computes the mean across the vector
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    # Get all summary
    summ = tf.summary.merge_all()

    # Run model
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(graph_dir)
    writer.add_graph(sess.graph)

    data_reader = dr.DataReader(sess,
                                train_dir,
                                test_dir,
                                INPUT_IMAGE_DIMENSION,
                                OUTPUT_VECTOR_SIZE,
                                INPUT_IMAGE_CHANNELS)

    coord = tf.train.Coordinator()

    # Train the model
    for i in range(STEP_SIZE):
        images, labels = data_reader.get_train_batch(coord, BATCH_SIZE)

        if i % 5 == 0:
            [_, s] = sess.run([accuracy, summ], feed_dict={x: images, y: labels})
            writer.add_summary(s, i)

        # Run the training step
        sess.run(train_step, feed_dict={x: images, y: labels})

    # For model testing
    with tf.name_scope("test_accuracy"):
        test_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        test_accuracy = tf.reduce_mean(tf.cast(test_prediction, tf.float32))
        test_accuracy_summary = tf.summary.scalar("test accuracy", test_accuracy)

    # Test the model
    for i in range(TEST_STEP_SIZE):
        test_images, test_labels = data_reader.get_train_batch(coord, BATCH_SIZE)
        if i % 5 == 0:
            [_, s] = sess.run([test_accuracy, test_accuracy_summary], feed_dict={x: test_images, y: test_labels})
            writer.add_summary(s, i)
        sess.run(logits, feed_dict={x: test_images, y: test_labels})

    coord.request_stop()
