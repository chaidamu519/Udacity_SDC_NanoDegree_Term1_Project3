
import pandas as pd
import numpy as np
## Read driving_log as Dataframe
samples = pd.read_csv('./driving_log.csv',  header=None)

## Provide the names of the columns
samples.columns = ['Center_image', 'Left_image', 'Right_image', 'steering angle', 'Throttle', 'Break', 'Speed']

## drop Throttle, Break and Speed columns
samples = samples.drop(['Throttle', 'Break', 'Speed'], axis = 1)


correction = 0.2
sample_center = samples.drop(['Left_image', 'Right_image'], axis = 1).rename(index=str, columns={"Center_image": "img_path", "steering angle": "angles"})
sample_left = samples.drop(['Center_image', 'Right_image'], axis = 1).rename(index=str, columns={"Left_image": "img_path", "steering angle": "angles"})
sample_right = samples.drop(['Center_image', 'Left_image'], axis = 1).rename(index=str, columns={"Right_image": "img_path", "steering angle": "angles"})
sample_left['angles'] += correction
sample_right['angles'] -= correction
data_set = pd.concat([sample_center, sample_left, sample_right], ignore_index=True)



from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(data_set, test_size=0.2)



import cv2
import random
## Change brightness
def change_brightness(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Convert to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    random_brightness_coefficient = np.random.uniform()+0.5 ## random ratio
    image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient 
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Convert to RGB
    return image_RGB

## Horizontal Flipping
def add_flip(image):
    return cv2.flip(image, 1 )


## Model

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, Dropout
from sklearn.utils import shuffle
from keras.models import Model
import matplotlib.pyplot as plt

def generator(samples, batch_size=32, augmentation = True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for ii in range(len(batch_samples) - 1):
                name = './Datas/data/IMG/'+batch_samples.iloc[ii][0].split('/')[-1]
                image = cv2.imread(name)
                angle = float(batch_samples.iloc[ii][1])
                images.append(image)
                angles.append(angle)
                
                if augmentation == True:
                    
                    ## Random change brightness 
                    image_clear = change_brightness(image)
                    angle_clear = angle
                    images.append(image_clear)
                    angles.append(angle_clear)
                    
                    ## Horizontal flipping
                    image_flip = add_flip(image)
                    angle_flip = - angle
                    images.append(image_flip)
                    angles.append(angle_flip)
                    
                                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32, augmentation = False)

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row, col, ch)))

## Model Architecture
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(3, 65, 320), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')

history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples),
validation_data=validation_generator, nb_val_samples = len(validation_samples), nb_epoch=10, verbose = 1)


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')

