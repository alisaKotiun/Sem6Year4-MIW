from keras import layers
from keras import models
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow import keras


def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(4, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.Conv2DTranspose(8, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.Conv2DTranspose(16, (3, 3), padding='same', activation='relu'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy')
    model.summary()
    return model


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

model = create_model(input_shape=(28, 28, 1))
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(train_images, train_images, epochs = 3, batch_size=64) # bigger number of epochs 
model.save_weights('wagi')


model1 = create_model(input_shape=(28, 28, 1)) # here we remove decoder (ukryta2 -> ukryta1 -> wejscia)
model1.load_weights('wagi')
model1.pop()
model1.pop()
model1.pop()
model1.pop()
model1.summary()

wynik1 = model1.predict(train_images[:500])

print(wynik1.shape)
print('wynik1 = {}'.format(wynik1.shape))
a,b,c,d = wynik1.shape
kod = wynik1.reshape(a, b*c*d)
print('kod = {}'.format(kod.shape))
print(kod)

# add kmeans  (print centrum)

(X_train, y_train), (_, _) = mnist.load_data()
y_pred = model1.predict(X_train)

model = KMeans(n_clusters=10)
pred = model.fit_predict(kod)

pred_labels = {}
for p in pred:
    if str(p) in pred_labels.keys():
        pred_labels[str(p)] += 1
    else:
        pred_labels[str(p)] = 1

train_labels = {}
for p in y_train:
    if str(p) in train_labels.keys():
        train_labels[str(p)] += 1
    else:
        train_labels[str(p)] = 1

for p, l in zip(pred_labels.items(), train_labels.items()):
    if l[1] < p[1]:
        print(f'{p[0]}: {int(l[1])/int(p[1])}')
    else:
        print(f'{p[0]}: {int(p[1])/int(l[1])}')

