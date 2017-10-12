import tensorflow as tf


def visualize_conv(input_images, image_dimension, input_channel, name):
    # Slice the first image of the convolutional result
    v = tf.slice(input_images, (0, 0, 0, 0), (1, -1, -1, -1))
    # Reshape the image to discard the first dimension
    v = tf.reshape(v, (image_dimension, image_dimension, input_channel))
    # Transpose the image to create 32 images
    v = tf.transpose(v, (2, 0, 1))
    # Reshape to create a 32 4D images with 1 channel
    v = tf.reshape(v, (-1, image_dimension, image_dimension, 1))
    tf.summary.image(name, v, input_channel)
