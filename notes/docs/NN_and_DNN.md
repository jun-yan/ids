---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Neural Network and Deep Learning

Neural Networks are described as a mathematical function that maps a given input to a desired output

Neural Networks consist of the following components
- An input layer, x
- An arbitrary amount of hidden layers
- An output layer, ŷ
- A set of weights and biases between each layer, W and b
- A choice of activation function for each hidden layer, σ. In this tutorial, we’ll use a Sigmoid activation function.

<img alt="" class="kq zh ei ev er fa v c" width="300" height="20" role="presentation" src="https://miro.medium.com/max/750/1*sX6T0Y4aa3ARh7IBS_sdqw.png" srcset="https://miro.medium.com/max/414/1*sX6T0Y4aa3ARh7IBS_sdqw.png 276w, https://miro.medium.com/max/750/1*sX6T0Y4aa3ARh7IBS_sdqw.png 500w" sizes="500px" title="Architecture of a 2-layer Neural Network" style=""> (Image by James Loy)

Each iteration of the training process consists of the following steps:
- Calculating the predicted output ŷ, known as **feedforward**
- Updating the weights and biases, known as **backpropagation**

Our goal in training is to find the best set of weights and biases that minimizes the loss function.

## Application

**scikit-learn** is one library that has neural network. In addition,deep Learning libraries such as **TensorFlow and Keras** also could build neural networks.

### Multi-layer Perceptron

<img alt="" width="300" height="20" src="https://scikit-learn.org/stable/_images/multilayerperceptron_network.png" style="width: 100px; height: 100px;" title=""> (Image by Scikit-learn)

#### [Mathematical Formulation](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#mathematical-formulation)
#### [Algorithms](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#algorithms)
#### [Choices of Loss Function](https://scikit-learn.org/stable/modules/sgd.html#mathematical-formulation)

class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000) [details](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn-neural-network-mlpclassifier)


```python
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
```


```python
iris_data = load_iris()
X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = iris_data.target
```


```python
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size=0.2)
sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(X_train)
X_testscaled=sc_X.transform(X_test)
```


```python
clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(X_trainscaled, y_train)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))
print(classification_report(y_test,y_pred))
```

    0.9666666666666667
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        11
               1       1.00      0.92      0.96        13
               2       0.86      1.00      0.92         6
    
        accuracy                           0.97        30
       macro avg       0.95      0.97      0.96        30
    weighted avg       0.97      0.97      0.97        30
    
    

### Deep learning

There are many deep learning libraries out there, but the most popular ones are **TensorFlow, Keras, and PyTorch**.

#### Relationship between ANN and DNN

<img alt="" class="v kh ki" src="https://miro.medium.com/max/1500/1*cuTSPlTq0a_327iTPJyD-Q.png" width="500" height="750" role="presentation">

A deep neural network (DNN) is an artificial neural network (ANN) with multiple layers between the input and output layers.

Deep Neural Networks (DNNs)  
Deep Neural Networks (DNNs) are typically Feed Forward Networks (FFNNs) in which data flows from the input layer to the output layer without going backward and the links between the layers are one way which is in the forward direction and they never touch a node again

A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN)

Recurrent Neural Network (RNN)  
A Recurrent Neural Network (RNN) addresses this issue which is a FFNN with a time twist.

LSTMs are a special kind of RNN, capable of learning long-term dependencies which make RNN smart at remembering things that have happened in the past and finding patterns across time to make its next guesses make sense.

Convolutional Neural Network (CNN, or ConvNet) is a class of deep neural networks which is most commonly applied to analyzing visual imagery.

<img class="TitleImage" src="https://pic2.zhimg.com/v2-f91e88cee5b0d44012a67da8853a8aa2_1440w.jpg?source=172ae18b" alt="卷积神经网络（CNN）基础及经典模型介绍" width="400" height="600" title="" style=""> (Image By Zhihu)

[Details](https://keras.io/api/layers)  
**Cov2d**:This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.   
**LeakyReLU**: Leaky version of a Rectified Linear Unit  
**MaxPooling2D**: Downsamples the input along its spatial dimensions (height and width)  
by taking the maximum value over an input window (of size defined by pool_size) for  
each channel of the input. The window is shifted by strides along each dimension.  
**Flatten**: Flattens the input. Does not affect the batch size.  
**Dense**: implements the operation: output = activation(dot(input, kernel) + bias)  
where activation is the element-wise activation function passed as the activation argument,  
kernel is a weights matrix created by the layer, and bias is a bias vector created  
by the layer (only applicable if use_bias is True). These are all attributes of Dense.


```python
# !pip install keras
from keras.datasets import fashion_mnist
(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()
```


```python
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow import keras
```


```python
print('Training data shape : ', train_X.shape, train_Y.shape)

print('Testing data shape : ', test_X.shape, test_Y.shape)
```

    Training data shape :  (60000, 28, 28) (60000,)
    Testing data shape :  (10000, 28, 28) (10000,)
    


```python
# Find the unique numbers from the train labels
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
```

    Total number of outputs :  10
    Output classes :  [0 1 2 3 4 5 6 7 8 9]
    


```python
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))
```




    Text(0.5, 1.0, 'Ground Truth : 9')




    
![png](output_32_1.png)
    



```python
train_X = train_X.reshape(-1, 28,28, 1) # to make a transform from 784 vectors to a 28*28 matrix
test_X = test_X.reshape(-1, 28,28, 1) # why add 1 in the end ???
train_X.shape, test_X.shape
```




    ((60000, 28, 28, 1), (10000, 28, 28, 1))




```python
train_X = train_X.astype('float32') # scaling
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.
```


```python
# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])
```

    Original label: 9
    After conversion to one-hot: [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
    


```python
from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
```


```python
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape
```




    ((48000, 28, 28, 1), (12000, 28, 28, 1), (48000, 10), (12000, 10))




```python
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
```


```python
batch_size = 64
epochs = 20
num_classes = 10
```


```python
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax'))
```


```python
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
```


```python
fashion_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 28, 28, 32)        320       
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 28, 28, 32)        0         
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        18496     
    _________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)    (None, 14, 14, 64)        0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 7, 7, 128)         73856     
    _________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)    (None, 7, 7, 128)         0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 4, 4, 128)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 2048)              0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               262272    
    _________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)    (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 356,234
    Trainable params: 356,234
    Non-trainable params: 0
    _________________________________________________________________
    


```python
fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
```

    Epoch 1/20
    750/750 [==============================] - 62s 82ms/step - loss: 0.4700 - accuracy: 0.8287 - val_loss: 0.3532 - val_accuracy: 0.8729
    Epoch 2/20
    750/750 [==============================] - 64s 86ms/step - loss: 0.2930 - accuracy: 0.8932 - val_loss: 0.2847 - val_accuracy: 0.8972
    Epoch 3/20
    750/750 [==============================] - 64s 85ms/step - loss: 0.2465 - accuracy: 0.9098 - val_loss: 0.2518 - val_accuracy: 0.9074
    Epoch 4/20
    750/750 [==============================] - 65s 87ms/step - loss: 0.2104 - accuracy: 0.9216 - val_loss: 0.2408 - val_accuracy: 0.9103
    Epoch 5/20
    750/750 [==============================] - 65s 87ms/step - loss: 0.1869 - accuracy: 0.9309 - val_loss: 0.2270 - val_accuracy: 0.9181
    Epoch 6/20
    750/750 [==============================] - 75s 100ms/step - loss: 0.1623 - accuracy: 0.9395 - val_loss: 0.2180 - val_accuracy: 0.9222
    Epoch 7/20
    750/750 [==============================] - 85s 113ms/step - loss: 0.1398 - accuracy: 0.9478 - val_loss: 0.2312 - val_accuracy: 0.9205
    Epoch 8/20
    750/750 [==============================] - 82s 109ms/step - loss: 0.1237 - accuracy: 0.9541 - val_loss: 0.2303 - val_accuracy: 0.9209
    Epoch 9/20
    750/750 [==============================] - 72s 96ms/step - loss: 0.1030 - accuracy: 0.9615 - val_loss: 0.2486 - val_accuracy: 0.9209
    Epoch 10/20
    750/750 [==============================] - 81s 108ms/step - loss: 0.0892 - accuracy: 0.9670 - val_loss: 0.2812 - val_accuracy: 0.9151
    Epoch 11/20
    750/750 [==============================] - 86s 115ms/step - loss: 0.0754 - accuracy: 0.9720 - val_loss: 0.2910 - val_accuracy: 0.9178
    Epoch 12/20
    750/750 [==============================] - 68s 90ms/step - loss: 0.0659 - accuracy: 0.9757 - val_loss: 0.2838 - val_accuracy: 0.9237
    Epoch 13/20
    750/750 [==============================] - 71s 94ms/step - loss: 0.0531 - accuracy: 0.9799 - val_loss: 0.3595 - val_accuracy: 0.9145
    Epoch 14/20
    750/750 [==============================] - 70s 94ms/step - loss: 0.0531 - accuracy: 0.9800 - val_loss: 0.3326 - val_accuracy: 0.9214
    Epoch 15/20
    750/750 [==============================] - 67s 90ms/step - loss: 0.0424 - accuracy: 0.9838 - val_loss: 0.3964 - val_accuracy: 0.9138
    Epoch 16/20
    750/750 [==============================] - 69s 92ms/step - loss: 0.0407 - accuracy: 0.9845 - val_loss: 0.3680 - val_accuracy: 0.9185
    Epoch 17/20
    750/750 [==============================] - 67s 89ms/step - loss: 0.0370 - accuracy: 0.9864 - val_loss: 0.4085 - val_accuracy: 0.9153
    Epoch 18/20
    750/750 [==============================] - 67s 89ms/step - loss: 0.0328 - accuracy: 0.9878 - val_loss: 0.3983 - val_accuracy: 0.9203
    Epoch 19/20
    750/750 [==============================] - 67s 89ms/step - loss: 0.0327 - accuracy: 0.9877 - val_loss: 0.4183 - val_accuracy: 0.9197
    Epoch 20/20
    750/750 [==============================] - 67s 89ms/step - loss: 0.0256 - accuracy: 0.9904 - val_loss: 0.4582 - val_accuracy: 0.9187
    


```python
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
```

    Test loss: 0.48407137393951416
    Test accuracy: 0.916700005531311
    


```python
accuracy = fashion_train.history['accuracy']
val_accuracy = fashion_train.history['val_accuracy']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
```


    
![png](output_45_0.png)
    



    
![png](output_45_1.png)
    



```python
predicted_classes = fashion_model.predict(test_X)
```


```python
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, test_Y.shape
correct = np.where(predicted_classes==test_Y)[0]
print ("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()
```

    Found 9159 correct labels
    


    
![png](output_47_1.png)
    


#### Unsupervised Pre-trained Neural Networks

##### Deep Generative Models

Boltzmann Machines/Deep Belief Neural Networks/Generative Adversarial Networks

##### Auto-encoder

#### Extension

Andrew Yan-Tak Ng   
popular courses on Coursera are Ng's: Machine Learning (#1), AI for Everyone, (#5), Neural Networks and Deep Learning (#6)
