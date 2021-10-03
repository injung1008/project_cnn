# LOAD LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#넘파이, 판다스, MATPLOT은 기본으로 항상

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm.keras import TqdmCallback

import warnings
warnings.filterwarnings('ignore')
# 지저분하게 워닝뜨는걸 막아준다.

x_train = np.load('D:\pig_project\\npy\_train_pig_x.npy')
y_train = np.load('D:\pig_project\\npy\_train_pig_y.npy')

x_test = np.load('D:\pig_project\\npy\_test_pig_x.npy')
y_test = np.load('D:\pig_project\\npy\_test_pig_y.npy')

fig = plt.figure(figsize=(10,10))

# for i in range(1):
#     i += 1
#     plt.subplot(1,1,i)
#     plt.imshow(x_train[i])
#     plt.axis('off')
# plt.show()


model = Sequential()

model.add(Conv2D(32,kernel_size=3,activation= 'relu', input_shape = (150,150,3) ))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation= 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(12,kernel_size=4,activation= 'relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(3,activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss = "sparse_categorical_crossentropy", metrics=['acc'])

# # 콜벡은 이렇게 선언해서 callbacks에 담아놓자
# earlyStopping = EarlyStopping(patience=100, verbose=0)
# reduce_lr_loss = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=0)
# tqdm = TqdmCallback(verbose=0) #진행율 표시해준다.(없으면 답답하다)
# callbacks = [earlyStopping, reduce_lr_loss, tqdm]

# hist = model.fit(x_train,y_train,
#                               epochs = 300,
#                               steps_per_epoch = 20,
#                               validation_split=0.2,
#                               callbacks=callbacks,
#                               verbose=0)


# print('train_acc:{0:.5f} , val_acc:{1:.5f}'.format(max(hist.history['acc']),max(hist.history['val_acc'])))


# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']


# e_loss = model.evaluate(x_test, y_test)
# print('e_loss : ', e_loss )

# #print('acc 전체 : ', acc)

# print('acc : ', acc[-1])
# print('loss : ', loss[-1])
# print('val acc : ', val_acc[-1])
# print('val loss : ', val_los[-1])

# import tensorflow as tf


# temp = model.predict(x_test)
# print('원본 : ', temp)
# temp = tf.argmax(temp, axis=1)
# temp = pd.DataFrame(temp)
# print('예측값 : ', temp)
# print('원래값 : ',y_test[:5])






# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title('Accuracy', fontsize=14)
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('Accuracy',fontsize=14)
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
