import tensorflow as tf

from MNIST import MNIST


# MNIST trainer class
# training with CNN model and write log..
# can use another model after adding in MNIST class
class MNISTTrainer(MNIST):
    train_op = None
    summary = None
    writer = None
    test_feed_dict = None

    def __init__(self, data_path=None, model_path=None, log_path=None):
        MNIST.__init__(self, model_path, data_path)

        self.log_path = log_path

        if data_path is not None:
            self.load_training_data(data_path)

    def init_log(self):
        if self.log_path is not None:
            X = self.mnist.test.images.reshape(-1, 28, 28, 1)
            Y = self.mnist.test.labels
            self.test_feed_dict = self.build_feed_dict(X, Y)

            self.summary = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)

    def add_log(self, name, graph, type='histogram'):
        if self.log_path is not None:
            if type == 'scalar':
                tf.summary.scalar(name, graph)
            else:
                tf.summary.histogram(name, graph)

    def write_log(self, epoch):
        if self.log_path is not None:
            summary = self.sess.run(self.summary, feed_dict=self.test_feed_dict)
            self.writer.add_summary(summary, epoch)

    def print_accuracy(self, epoch):
        if self.log_path is not None:
            accuracy = self.check_accuracy(self.test_feed_dict)
            print 'Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy

    def build_training_op(self, learning_rate, decay):
        with tf.name_scope('cost') as scope:
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.model, self.Y))

        self.add_log('Y', self.Y)
        self.add_log('cost', cost, 'scalar')

        self.train_op = tf.train.RMSPropOptimizer(learning_rate, decay).minimize(cost)

    def training_once(self, batch_size, p_keep_conv, p_keep_hidden):
        total_batch = int(self.mnist.train.num_examples/batch_size)

        for step in range(total_batch):
            batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape(-1, 28, 28, 1)
            feed_dict = self.build_feed_dict(batch_xs, batch_ys, p_keep_conv, p_keep_hidden)

            self.sess.run(self.train_op, feed_dict=feed_dict)

    # training several times
    def training(self,
                 learning_rate=0.001,
                 decay=0.9,
                 training_epochs=15,
                 batch_size=100,
                 p_keep_conv=1.,
                 p_keep_hidden=1.):

        self.print_status('Building CNN model..')

        self.build_cnn_model(p_keep_conv, p_keep_hidden)

        self.build_training_op(learning_rate, decay)

        self.print_status('Start training. Please be patient. :-)')

        self.init_session()

        # init summary for tensorboard
        self.init_log()

        # start training
        for epoch in range(training_epochs):
            self.training_once(batch_size, p_keep_conv, p_keep_hidden)

            self.write_log(epoch)

            self.print_accuracy(epoch)

        # TODO: save the best model only
        self.save_model()

        self.print_status('Learning Finished!')
