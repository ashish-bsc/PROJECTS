from keras.layers import Dense
from keras.models import Sequential
#from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

classifier = Sequential()
#first convolutional layer
classifier.add(Conv2D(32,3,3,activation='relu', input_shape=(64,64,3)))

#maxpooling of first convolutional layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#2n convolutional layer
classifier.add(Conv2D(32,3,3,activation='relu'))

#mac pooling of 2nd convolutiinal layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#flatten our layers
classifier.add(Flatten())

#adding hidden layer
classifier.add(Dense(units=128, activation='relu'))

#adding output layer
classifier.add(Dense(units=1,activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train = ImageDataGenerator(rescale=1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test = ImageDataGenerator(rescale=1/255)

training_set = train.flow_from_directory('dataset/train', target_size=(64,64), batch_size=32, class_mode='binary')

test_set = test.flow_from_directory('dataset/test', target_size=(64,64), batch_size=32, class_mode='binary')

classifier.fit_generator(training_set,nb_val_samples=2000, nb_epoch=10, validation_data=test_set, steps_per_epoch=20)

classifier.save('model.h5')

classifier.load_weights('model.h5')

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/cat_1.png', target_size=(64,64))
test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Dog'
else:
    prediction = 'Cat'