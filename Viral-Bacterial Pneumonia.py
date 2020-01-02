from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

#for viral or bacterial pneumonia
classifier = Sequential()

classifier.add(Conv2D(32,3,3, activation='relu', input_shape=(64,64,3)))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(32,3,3, activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units=100, activation='relu'))

classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train = ImageDataGenerator(rescale=1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test = ImageDataGenerator(rescale=1/255)

training_set = train.flow_from_directory('dataset-1/train', target_size = (64,64), batch_size = 32, class_mode = 'binary')

test_set = test.flow_from_directory('dataset-1/test', target_size = (64,64), batch_size = 32, class_mode = 'binary')

classifier.fit_generator(training_set, nb_val_samples = 2000, nb_epoch = 2, validation_data = test_set, steps_per_epoch=20)

classifier.save('model-1.h5')

classifier.load_weights('model-1.h5')

import numpy as np
from keras.preprocessing import image

#code for bacterial or viral chest
    
test_image = image.load_img('person80_virus_150.jpeg', target_size=(64,64))
test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Virus'
else:
    prediction = 'Bacteria'