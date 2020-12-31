import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, Dropout, Dense, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import time
import pickle


#### Global settings ####
path = 'Data'
test_ratio = .2
valid_ratio = .2
batch_size = 50
epochs = 10
steps = 2000
start_time = time.time()


#### Setup data ####
images = []
labels = []
my_list = os.listdir(path)
class_num = len(my_list)
samples = []

print('Importing classes: ')
for i in range(0, class_num):
    pic_list = os.listdir(path+'/'+str(i))
    for j in pic_list:
        cur_img = cv2.imread(path+'/'+str(i)+'/'+j)
        cur_img = cv2.resize(cur_img, (32, 32))
        images.append(cur_img)
        labels.append(i)
    print(i, end=' ')
print('\n\n')

images = np.array(images)
labels = np.array(labels)


#### split da data ####
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_ratio)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_ratio)

print('Train data: ', X_train.shape, " : ", y_train.shape)
print('Valid data: ', X_valid.shape, " : ", y_valid.shape)
print('Test data:  ', X_test.shape, " : ", y_test.shape)

for i in range(0, class_num):
    samples.append(len(np.where(y_train==i)[0]))


#### preprocessing ####
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_valid = np.array(list(map(preprocessing, X_valid)))
X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(X_train.shape[0],
                          X_train.shape[1],
                          X_train.shape[2], 1)

X_valid = X_valid.reshape(X_valid.shape[0],
                          X_valid.shape[1],
                          X_valid.shape[2], 1)

X_test = X_test.reshape(X_test.shape[0],
                          X_test.shape[1],
                          X_test.shape[2], 1)

img_gen = ImageDataGenerator(width_shift_range=.1,
                             height_shift_range=.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

img_gen.fit(X_train)

y_train = to_categorical(y_train, class_num)
y_valid = to_categorical(y_valid, class_num)
y_test = to_categorical(y_test, class_num)


#### Create CNN ####
def cnn():
    filters = 60
    filters_size1 = (5, 5)
    filters_size2 = (3, 3)
    pool_size = (2, 2)
    nodes = 500

    model = Sequential()
    model.add((Conv2D(filters, filters_size1, input_shape=(32, 32, 1), activation='relu')))
    model.add(((Conv2D(filters, filters_size1, activation='relu'))))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(((Conv2D(filters//2, filters_size2, activation='relu'))))
    model.add(((Conv2D(filters//2, filters_size2, activation='relu'))))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(.5))
    model.add(Flatten())
    model.add(Dense(nodes, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(class_num, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = cnn()

history = model.fit(img_gen.flow(X_train,
                       y_train,
                       batch_size=batch_size),
          steps_per_epoch= steps,
          epochs = epochs,
          validation_data=(X_valid, y_valid),
          shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epochs')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('accuracy')
plt.xlabel('epochs')

plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])

pickle.dump(model, open('model_train.p', 'wb'))

end_time = time.time()
print('\n learning time: ', end_time-start_time)