import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from TFUtils import TFUtils


# MNIST base class
# main purpose is building cnn model
# can add other models
class MNIST:
    model_path = None
    data_path = None

    sess = None
    model = None
    mnist = None

    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])

    p_keep_conv = tf.placeholder(tf.float32)
    p_keep_hidden = tf.placeholder(tf.float32)

    def __init__(self, model_path=None, data_path=None):
        self.model_path = model_path
        self.data_path = data_path

    def init_session(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def print_status(self, text):
        print '---'
        print text

    def load_training_data(self, data_path):
        print 'Preparing MNIST data..'

        self.mnist = input_data.read_data_sets(data_path, one_hot=True)

    def build_feed_dict(self, X, Y, p_keep_conv=1., p_keep_hidden=1.):
        return {
            self.X: X,
            self.Y: Y,
            self.p_keep_conv: p_keep_conv,
            self.p_keep_hidden: p_keep_hidden
        }

    # define model
    def build_cnn_model(self, p_keep_conv=1., p_keep_hidden=1.):
        W1 = TFUtils.xavier_init([3, 3, 1, 32], 'W1')
        W2 = TFUtils.xavier_init([3, 3, 32, 64], 'W2')
        W3 = TFUtils.xavier_init([3, 3, 64, 128], 'W3')
        W4 = TFUtils.xavier_init([128 * 4 * 4, 625], 'W4')
        W5 = TFUtils.xavier_init([625, 10], 'W5')

        with tf.name_scope('layer1') as scope:
            # L1 Conv shape=(?, 28, 28, 32)
            #    Pool     ->(?, 14, 14, 32)
            L1 = TFUtils.build_cnn_layer(self.X, W1, p_keep_conv)
        with tf.name_scope('layer2') as scope:
            # L2 Conv shape=(?, 14, 14, 64)
            #    Pool     ->(?, 7, 7, 64)
            L2 = TFUtils.build_cnn_layer(L1, W2, p_keep_conv)
        with tf.name_scope('layer3') as scope:
            # L3 Conv shape=(?, 7, 7, 128)
            #    Pool     ->(?, 4, 4, 128)
            #    Reshape  ->(?, 625)
            reshape = [-1, W4.get_shape().as_list()[0]]
            L3 = TFUtils.build_cnn_layer(L2, W3, p_keep_conv, reshape=reshape)
        with tf.name_scope('layer4') as scope:
            # L4 FC 4x4x128 inputs -> 625 outputs
            L4 = tf.nn.relu(tf.matmul(L3, W4))
            L4 = tf.nn.dropout(L4, p_keep_hidden)

        # Output(labels) FC 625 inputs -> 10 outputs
        self.model = tf.matmul(L4, W5, name='model')

        return self.model

    def save_model(self):
        if self.model_path is not None:
            self.print_status('Saving my model..')

            saver = tf.train.Saver(tf.global_variables())
            saver.save(self.sess, self.model_path)

    def load_model(self):
        self.build_cnn_model()

        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

    def check_accuracy(self, test_feed_dict=None):
        check_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
        accuracy_rates = self.sess.run(accuracy, feed_dict=test_feed_dict)

        return accuracy_rates
