---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
# Recurrent Neural Network

- DLP/NN: deal
- CNN: capture the spatial data structure(image analysis)
- RNN: capture the sequential data structure(sentence, stock price)

**Example** : "I like eating apple." and "Apple is a company.". If we want to specify different meanings of a word, we need to take the nearby words into consideration.

How does RNN works:\
The hidden layer remembers the infomation of the previous hidden layer $h_{t-1}$, and then learn from the current data $X_t$.
![Recurrent_neural_network_unfold.svg](attachment:Recurrent_neural_network_unfold.svg)
$$
O_t=g(Wh_t)\\
h_t=f(UX_t+Vh_{t-1})
$$
- $X_t$: input vector
- $h_t$: hidden layer vector
- $O_t$: output vector
- $W,U,V$: parameter matrices

**Example** Assume we have trained a RNN, 2 nodes with weights $W,U,V=(0.5,0.5)'$, our sequence 
$(1,1)',(1,2)',...$\
$h_1=(0.5*1+0.5*1,0.5*1+0.5*1)=(1,1),O_1=(1,1)$\
$h_2=(0.5*1+0.5*2,0.5*1+0.5*2)+(1,1)=(2.5,2.5),O_2=(2.5,2.5)$

LSTM: "long short term memory", a commonly used RNN model.


```python
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])

# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
```

    [[16, 1], [46, 17], [5, 10], [35, 17], [27], [2], [4, 10], [40, 46], [4, 17], [26, 41, 1, 6]]
    


```python
e = Embedding(200, 32, input_length=50)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
```

    [[16  1  0  0]
     [46 17  0  0]
     [ 5 10  0  0]
     [35 17  0  0]
     [27  0  0  0]
     [ 2  0  0  0]
     [ 4 10  0  0]
     [40 46  0  0]
     [ 4 17  0  0]
     [26 41  1  6]]
    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding_1 (Embedding)     (None, 4, 8)              400       
                                                                     
     flatten_1 (Flatten)         (None, 32)                0         
                                                                     
     dense_2 (Dense)             (None, 1)                 33        
                                                                     
    =================================================================
    Total params: 433
    Trainable params: 433
    Non-trainable params: 0
    _________________________________________________________________
    None
    Accuracy: 89.999998
    

# Convolutional Neural Network

## Filter
Filter is also known as kernel. It is a designed matrix in CNN, which extracts the local features from a data.

## Basic concepts
- Padding: Addition of (typically) 0-valued pixels on the borders of an image
- Pooling: Reduce the dimensions of data by combining the outputs of previous neuron into a single neuron in the next layer.
- Channels: Number of filters in a layer



```python
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
```

    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    170500096/170498071 [==============================] - 123s 1us/step
    170508288/170498071 [==============================] - 123s 1us/step
    


```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 30, 30, 32)        896       
                                                                     
     max_pooling2d (MaxPooling2D  (None, 15, 15, 32)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 6, 6, 64)         0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928     
                                                                     
    =================================================================
    Total params: 56,320
    Trainable params: 56,320
    Non-trainable params: 0
    _________________________________________________________________
    

The input data is $32*32*3$ tensor, and the layers are:
- Conv layer: $3*3$ filter, channels=32
- Pooling layer: $2*2$ maxpooling
- Conv layer: $3*3$ filter, channels=64
- Pooling layer: $2*2$ maxpooling
- Conv layer: $3*3$ filter, channels=64


```python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 30, 30, 32)        896       
                                                                     
     max_pooling2d (MaxPooling2D  (None, 15, 15, 32)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 6, 6, 64)         0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928     
                                                                     
     flatten (Flatten)           (None, 1024)              0         
                                                                     
     dense (Dense)               (None, 64)                65600     
                                                                     
     dense_1 (Dense)             (None, 10)                650       
                                                                     
    =================================================================
    Total params: 122,570
    Trainable params: 122,570
    Non-trainable params: 0
    _________________________________________________________________
    

To complete model, the last layer(dense layer) outputs the 1-dimensional vector to make a classification.

[See the picture here](https://www.google.com/url?sa=i&url=https%3A%2F%2Ftowardsdatascience.com%2Fa-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53&psig=AOvVaw0X2854QLmfCfvXO8Lj5eC9&ust=1637352332958000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCID77_naovQCFQAAAAAdAAAAABAD)


```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
```

    Epoch 1/10
    1563/1563 [==============================] - 59s 37ms/step - loss: 1.5502 - accuracy: 0.4321 - val_loss: 1.3032 - val_accuracy: 0.5331
    Epoch 2/10
    1563/1563 [==============================] - 58s 37ms/step - loss: 1.2001 - accuracy: 0.5765 - val_loss: 1.1816 - val_accuracy: 0.5806
    Epoch 3/10
    1563/1563 [==============================] - 54s 34ms/step - loss: 1.0436 - accuracy: 0.6316 - val_loss: 1.0422 - val_accuracy: 0.6347
    Epoch 4/10
    1563/1563 [==============================] - 54s 35ms/step - loss: 0.9299 - accuracy: 0.6737 - val_loss: 0.9405 - val_accuracy: 0.6712
    Epoch 5/10
    1563/1563 [==============================] - 55s 35ms/step - loss: 0.8501 - accuracy: 0.7027 - val_loss: 0.9336 - val_accuracy: 0.6752
    Epoch 6/10
    1563/1563 [==============================] - 55s 35ms/step - loss: 0.7862 - accuracy: 0.7249 - val_loss: 0.8879 - val_accuracy: 0.6898
    Epoch 7/10
    1563/1563 [==============================] - 55s 35ms/step - loss: 0.7354 - accuracy: 0.7414 - val_loss: 0.8518 - val_accuracy: 0.7038
    Epoch 8/10
    1563/1563 [==============================] - 57s 36ms/step - loss: 0.6873 - accuracy: 0.7593 - val_loss: 0.8349 - val_accuracy: 0.7101
    Epoch 9/10
    1563/1563 [==============================] - 56s 36ms/step - loss: 0.6497 - accuracy: 0.7722 - val_loss: 0.8684 - val_accuracy: 0.7094
    Epoch 10/10
    1563/1563 [==============================] - 56s 36ms/step - loss: 0.6069 - accuracy: 0.7865 - val_loss: 0.8711 - val_accuracy: 0.7054
    


```python
print(test_acc)
```

    0.7053999900817871
    
