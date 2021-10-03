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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
# 지저분하게 워닝뜨는걸 막아준다.



import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. data 불러오기 

# train_datagen = ImageDataGenerator(rescale=1./255,
#                     # horizontal_flip=True,
#                     # vertical_flip=False,
#                     # width_shift_range=0.1,
#                     # height_shift_range=0.1,
#                     # rotation_range=5,
#                     # zoom_range=0.6,
#                     # shear_range=0.5,
#                     # fill_mode='nearest'
#                     )



# # rotation_range: 이미지 회전 범위 (degrees)
# # width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 (원본 가로, 세로 길이에 대한 비율 값)
# # rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
# # shear_range: 임의 전단 변환 (shearing transformation) 범위
# # zoom_range: 임의 확대/축소 범위
# # horizontal_flip: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
# # fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식



# test_datagen = ImageDataGenerator(rescale=1./255)

# xy_train = train_datagen.flow_from_directory(
#     'D:\\pig_project\\_data\\Training',    
#         target_size=(150,150), 
#         batch_size=11000,
#         class_mode='binary',    
#         shuffle=False)


# xy_test = test_datagen.flow_from_directory(
#         'D:\\pig_project\\_data\\Validation',
#         target_size=(150,150),
#         batch_size=1200,
#         class_mode='binary'
# )

# print(xy_train[0][0].shape, xy_train[0][1].shape)
# print(xy_test[0][0].shape, xy_test[0][1].shape)


# np.save('D:\\pig_project\\npy\\cow_train_pig_x.npy', arr=xy_train[0][0])
# np.save('D:\\pig_project\\npy\\cow_train_pig_y.npy', arr=xy_train[0][1])
# np.save('D:\\pig_project\\npy\\cow_test_pig_x.npy', arr=xy_test[0][0])
# np.save('D:\\pig_project\\npy\\cow_test_pig_y.npy', arr=xy_test[0][1])

# change_datagen = ImageDataGenerator(rescale=1./255,
#                     horizontal_flip=True,
#                     vertical_flip=False,
#                     # width_shift_range=0.2,
#                     # height_shift_range=0.1,
#                     rotation_range=5,
#                     zoom_range=0.6,
#                     # shear_range=0.5,
#                     fill_mode='nearest'
#                     )

x_train = np.load('D:\pig_project\\npy\\cow_train_pig_x.npy')
y_train = np.load('D:\pig_project\\npy\\cow_train_pig_y.npy')

x_test = np.load('D:\pig_project\\npy\\cow_test_pig_x.npy')
y_test = np.load('D:\pig_project\\npy\\cow_test_pig_y.npy')


#!#####################################################################################


# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

# print(x_train.shape) #(10230, 150, 150, 3)
# print(x_test.shape) #(1200, 150, 150, 3)
#! data 증폭 

# augment_size = 10000 #배치사이즈 - 증폭시킬 데이터 양 
# randidx = np.random.randint(x_train.shape[0], size=augment_size)

# # print(randidx) #[4812 2854 1653 ... 1292 3950 1941]    
# # print(randidx.shape) #(3000,)배치 사이즈만큼 랜덤하게 들어감 

# x_augmented = x_train[randidx].copy() #메모리가 공유되는걸 방지하기 위해 카피해서 진행.. 
# y_augmented = y_train[randidx].copy()
# # print(x_augmented.shape) #(3000, 150, 150, 3)
# # print(x_train.shape) #(5485, 150, 150, 3)
# # print(x_test.shape) #(685, 150, 150, 3)
# x_augmented = change_datagen.flow(x_augmented, np.zeros(augment_size),
#                                 batch_size=augment_size, 
#                                 #save_to_dir='D:\Study\\temp',
#                                 shuffle=False).next()[0] 
#                                 #x는 [0]에 있고 y는 [1]에 있어서 마지막에 [0]을 붙임으로서 x만 뽑아줌

# print(x_augmented.shape) #(3000, 150, 150, 3)


# x_train = np.concatenate((x_train, x_augmented))
# y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape) #(8485, 150, 150, 3)

# ##!증폭데이터 시각화 
# import matplotlib.pyplot as plt
# plt.figure(figsize=(2,2))
# for i in range(10):
#     plt.subplot(2, 10, i+1)
#     plt.axis('off')
#     plt.title('a')
#     plt.imshow(x_train[i], cmap='gray')

#     plt.subplot(2, 10, i+11)
#     plt.axis('off')
#     plt.title('b')

#     plt.imshow(x_augmented[i], cmap='gray')

# plt.show()


# from sklearn.model_selection import train_test_split
# x_test, x_val, y_test, y_val = train_test_split(x_test, y_test,
#                  train_size = 0.4, shuffle=True, random_state=66)
print(np.unique(y_train))#[0,1,2]

# fig = plt.figure(figsize=(10,10))

# for i in range(1):
#     i += 1
#     plt.subplot(1,1,i)
#     plt.imshow(x_train[i])
#     plt.axis('off')
# plt.show()


# include_top = False인 경우 input_shape = img_size 지정 필요
# conv에 대한 가중치 학습이 없음
import tensorflow as tf
VGG16_MODEL = tf.keras.applications.VGG16(input_shape = (150,150,3),
                                         include_top = False,
                                         weights = 'imagenet')

VGG16_MODEL.trainable=False
# flatten이 없음 ( globalaveragepooling으로 대체 )
#  ==> 가중치가 필요없음
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# FFNN의 가중치는 학습됨
dense_layer = tf.keras.layers.Dense(3, activation ='relu' )
prediction_layer = tf.keras.layers.Dense(3, activation ='softmax' )

model = tf.keras.Sequential([
    VGG16_MODEL,
    global_average_layer,
    dense_layer,
    prediction_layer
])
# model : vgg16이 갖고있는 가중치 + FFNN 가중치로 학습


model.summary()

print("[INFO] compiling model...")
# opt = Adam(lr=INIT_LR)
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',
	metrics=["acc"])


# earlyStopping = EarlyStopping(monitor='val_acc',patience=50, verbose=0)
# reduce_lr_loss = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=0)
# ReduceLROnPlateau 모델에 개선이 없을 경우. learning rate를 조절해 모델의 개선을 유도하는 콜백함수 

tqdm = TqdmCallback(verbose=0) #진행율 표시해준다.(없으면 답답하다)
# callbacks = [earlyStopping, reduce_lr_loss, tqdm]
callbacks = [tqdm]


hist = model.fit(x_train,y_train,
                              epochs = 20,
                              steps_per_epoch = 20,
                              validation_data=(x_test,y_test),
                              callbacks=callbacks,
                              batch_size=300,
                              verbose=0)


print('train_acc:{0:.5f} , val_acc:{1:.5f}'.format(max(hist.history['acc']),max(hist.history['val_acc'])))


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


e_loss = model.evaluate(x_test, y_test)


#print('acc 전체 : ', acc)

print('acc : ', acc)
print('loss : ', loss[-1])
print('val acc : ', val_acc)
print('val loss : ', val_loss[-1])

import tensorflow as tf


temp = model.predict(x_test)
print('원본 : ', temp[:10])
temp = tf.argmax(temp, axis=1)


temp = pd.DataFrame(temp)
print('예측값 : ', temp[:10])
print('원래값 : ',y_test[:10])




plt.style.use("ggplot")
plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('loss',fontsize=14)
plt.legend(['Train', 'val'], loc='upper left')
plt.show()


#! dense_layer = tf.keras.layers.Dense(3, activation ='relu' )
# prediction_layer = tf.keras.layers.Dense(3, activation ='softmax' )
# acc :  0.9583333134651184
# loss :  0.5334665179252625
# val acc :  0.9708333611488342
# val loss :  0.5072580575942993
# 원본 :  [[0.0815227  0.00158966 0.91688764]
#  [0.9169666  0.0205895  0.06244387]
#  [0.2812329  0.43425155 0.28451562]
#  ...
#  [0.2812329  0.43425155 0.28451562]
#  [0.2812329  0.43425155 0.28451562]
#  [0.2812329  0.43425155 0.28451562]]
# 예측값 :        0
# 0     2
# 1     0
# 2     1
# 3     2
# 4     1
# ...  ..
# 1195  2
# 1196  0
# 1197  1
# 1198  1
# 1199  1
# [1200 rows x 1 columns]
# 원래값 :  [2. 0. 1. 2. 1.]

