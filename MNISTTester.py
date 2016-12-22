import tensorflow as tf
import numpy as np
import os

from PIL import Image, ImageFilter
from random import randint
from matplotlib import pyplot as plt
from MNIST import MNIST


# MNIST Tester class
# check accuracy of test set
# predict random number from test set
# predict number from image
class MNISTTester(MNIST):
    def __init__(self, model_path=None, data_path=None):
        MNIST.__init__(self, model_path, data_path)

        self.init()

    def init(self):
        self.print_status('Loading a model..')

        self.init_session()

        self.load_model()

        if self.data_path is not None:
            self.load_training_data(self.data_path)

    def classify(self, feed_dict):
        number = self.sess.run(tf.argmax(self.model, 1), feed_dict)[0]
        accuracy = self.sess.run(tf.nn.softmax(self.model), feed_dict)[0]

        return number, accuracy[number]

    def accuracy_of_testset(self):
        self.print_status('Calculating accuracy of test set..')

        X = self.mnist.test.images.reshape(-1, 28, 28, 1)
        Y = self.mnist.test.labels
        test_feed_dict = self.build_feed_dict(X, Y)

        accuracy = self.check_accuracy(test_feed_dict)

        self.print_status('CNN accuracy of test set: %f' % accuracy)

    def predict_random(self, show_image=False):
        num = randint(0, self.mnist.test.images.shape[0])
        image = self.mnist.test.images[num]
        label = self.mnist.test.labels[num]

        feed_dict = self.build_feed_dict(image.reshape(-1, 28, 28, 1), [label])

        (number, accuracy) = self.classify(feed_dict)
        label = self.sess.run(tf.argmax(label, 0))

        self.print_status('Predict random item: %d is %d, accuracy: %f' %
                                              (label, number, accuracy))

        if show_image is True:
            plt.imshow(image.reshape(28, 28))
            plt.show()

    def predict(self, filename):
        data = self.load_image(filename)

        number, accuracy = self.classify({self.X: data})

        self.print_status('%d is %s, accuracy: %f' % (number, os.path.basename(filename), accuracy))

    def load_image(self, filename):
        img = Image.open(filename).convert('L')

        # resize to 28x28
        img = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

        # normalization : 255 RGB -> 0, 1
        data = [(255 - x) * 1.0 / 255.0 for x in list(img.getdata())]

        # reshape -> [-1, 28, 28, 1]
        return np.reshape(data, (-1, 28, 28, 1)).tolist()
