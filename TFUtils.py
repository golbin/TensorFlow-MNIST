import tensorflow as tf


# Utility packages for lazy me
class TFUtils:
    def __init__(self):
        return

    # Xavier initialization
    @staticmethod
    def xavier_init(shape, name='', uniform=True):
        num_input = sum(shape[:-1])
        num_output = shape[-1]

        if uniform:
            init_range = tf.sqrt(6.0 / (num_input + num_output))
            init_value = tf.random_uniform_initializer(-init_range, init_range)
        else:
            stddev = tf.sqrt(3.0 / (num_input + num_output))
            init_value = tf.truncated_normal_initializer(stddev=stddev)

        return tf.get_variable(name, shape=shape, initializer=init_value)

    @staticmethod
    def conv2d(X, W, strides=None, padding='SAME'):
        if strides is None:
            strides = [1, 1, 1, 1]

        return tf.nn.conv2d(X, W, strides=strides, padding=padding)

    @staticmethod
    def max_pool(X, ksize=None, strides=None, padding='SAME'):
        if ksize is None:
            ksize = [1, 2, 2, 1]

        if strides is None:
            strides = [1, 2, 2, 1]

        return tf.nn.max_pool(X, ksize=ksize, strides=strides, padding=padding)

    @staticmethod
    def build_cnn_layer(X, W, p_dropout=1., pool=True, reshape=None):
        L = tf.nn.relu(TFUtils.conv2d(X, W))

        if pool is True:
            L = TFUtils.max_pool(L)

        if reshape is not None:
            L = tf.reshape(L, reshape)

        if p_dropout == 1:
            return L
        else:
            return tf.nn.dropout(L, p_dropout)
