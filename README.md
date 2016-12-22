# MNIST with TensorFlow

- Just for training TensorFlow and Deep Learning
- Try to make easy to understand building layers and using TensorFlow
    - write summaries for TensorBoard
    - save and load a model and reuse for prediction
- Pre-trained model with default options is included
    - you can test prediction and TensorBoard without any hassle

## Class

- **MNIST** : building model (currently CNN only)
- **MNISTTrainer** : training logic and steps
- **MNISTTester** : test trained model and an image
- **TFUtils** : Xavier initialization and a small utilities for my laziness

## Excutable Scripts

- **train.py** : can use below options
    - learning_rate=0.001
    - decay=0.9
    - training_epochs=10
    - batch_size=100
    - p_keep_conv=0.8
    - p_keep_hidden=0.5
- **test.py**
    - prediction test with MNIST test set
    - prediction test with image file
        - only for square images and single number
        - size is not matter

## Results

```
➜  TensorFlow-MNIST# python train.py 
Preparing MNIST data..
Extracting mnist/data/train-images-idx3-ubyte.gz
Extracting mnist/data/train-labels-idx1-ubyte.gz
Extracting mnist/data/t10k-images-idx3-ubyte.gz
Extracting mnist/data/t10k-labels-idx1-ubyte.gz
---
Building CNN model..
---
Start training. Please be patient. :-)
Epoch: 0001 / Accuracy = 0.9511
Epoch: 0002 / Accuracy = 0.9634
...
---
Saving my model..
---
Learning Finished!
```

```
➜  TensorFlow-MNIST# python test.py
---
Loading a model..
Preparing MNIST data..
Extracting mnist/data/train-images-idx3-ubyte.gz
Extracting mnist/data/train-labels-idx1-ubyte.gz
Extracting mnist/data/t10k-images-idx3-ubyte.gz
Extracting mnist/data/t10k-labels-idx1-ubyte.gz
---
Calculating accuracy of test set..
---
CNN accuracy of test set: 0.993600
---
Predict random item: 5 is 5, accuracy: 1.000
---
4 is digit-4.png
---
2 is digit-2.png
---
5 is digit-5.png
```

```
➜  TensorFlow-MNIST# tensorboard --logdir=logs/mnist-cnn
```
