from keras.models import Sequential
from keras.layers import Dense
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten
from keras import optimizers
from keras.losses import categorical_crossentropy
from keras.initializers import he_normal
from keras import metrics

import numpy as np

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

## Data load

print ('---------------------------------')
#print (' Training Data : %d' % len(X_train))
#print (' Test Data : %d' % len(X_test))
print ('---------------------------------')

## Training
model = Sequential()

## Architecture
model.add(Conv2D( filters = 96, kernel_size = (11,11), strides = 4, padding = 'same', activation = 'relu', input_shape = (224, 224, 3), kernel_initializer = 'he_normal'))
model.add(MaxPooling2D( pool_size = (3,3), strides = (2,2), padding= 'same', data_format = None)) # overlapping pooling

model.add(Conv2D( filters = 256, kernel_size = (5,5), strides = 1, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
model.add(MaxPooling2D( pool_size = (3,3), strides = (2,2), padding= 'same', data_format = None)) 

model.add(Conv2D( filters = 384, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
model.add(Conv2D( filters = 384, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
model.add(Conv2D( filters = 256, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
model.add(MaxPooling2D( pool_size = (3,3), strides = (2,2), padding= 'same', data_format = None))

model.add(Flatten())
model.add(Dense( units = 4096, activation = 'relu'))
model.add(Dense( units = 4096, activation = 'relu'))

model.add(Dense( units = 1000, activation = 'softmax'))
model.summary()

## Optimizer
adam = optimizers.adam(lr=0.00005, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay= 0, amsgrad = False)
momentum = optimizers.SGD(lr=0.01, momentum = 0.9, decay=1e-6)

#model.compile(optimizer = adam, loss = categorical_crossentropy, metrics=[metrics.categorical_accuracy])

# when using the categorical_crossentropy loss, your targets should be in categorical format (one- hot encoding)

#model.fit(X_train, Y_train, batch_size = 256, epochs = 100, validation_data=(X_test, Y_test))
#score = model.evaluate(X_test, Y_test, batch_size = 64)