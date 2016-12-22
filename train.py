import os

from MNISTTrainer import MNISTTrainer


####################
# directory settings
script_dir = os.path.dirname(os.path.abspath(__file__))

data_path = script_dir + '/mnist/data/'
model_path = script_dir + '/models/mnist-cnn'
log_path = script_dir + '/logs/mnist-cnn'

##########
# training
mnist = MNISTTrainer(
            data_path=data_path,
            model_path=model_path,
            log_path=log_path)

mnist.training(
    learning_rate=0.001,
    decay=0.9,
    training_epochs=10,
    batch_size=100,
    p_keep_conv=0.8,
    p_keep_hidden=0.5)
