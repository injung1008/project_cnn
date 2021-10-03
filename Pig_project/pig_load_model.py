import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train = np.load('D:\pig_project\\npy\_train_pig_x.npy')
y_train = np.load('D:\pig_project\\npy\_train_pig_y.npy')

x_test = np.load('D:\pig_project\\npy\_test_pig_x.npy')
y_test = np.load('D:\pig_project\\npy\_test_pig_y.npy')

print(np.unique(y_train)) #[0. 1. 2.]


print(x_train.shape)#(8521, 150, 150, 3)
print(y_train.shape)#(8521)
print(x_test.shape)#(1065, 150, 150, 3)
print(y_test.shape)#(1065)


# #! 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(100,(2,2), input_shape=(300,300,3)))
model.add(Conv2D(10,(2,2), input_shape=(300,100,3)))

model.add(Conv2D(180, (2,2)))
model.add(Dense(10, activation='relu', input_shape=(300,300,3)))
model.add(Flatten())
model.add(Dense(160,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(6,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

# #3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=3) # 



#x y가 붙어있는경우 fit_generator사용하면 fit과 동일한 결과 나옴
hist = model.fit(x_train,y_train, epochs=10,batch_size=300,
                    validation_split=0.1)
                   # validation_steps=4) 
                    #validation_steps=4 이런 파라미터가 있다 


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


e_loss = model.evaluate(x_test, y_test)
print('e_loss : ', e_loss )

#print('acc 전체 : ', acc)

print('acc : ', acc[-1])
print('loss : ', loss)
print('val acc : ', val_acc)
print('val loss : ', val_loss)



temp = model.predict(x_test)
print('원본 : ', temp[:5])
# temp = tf.argmax(temp, axis=1)
# #temp = pd.DataFrame(temp)
# print('예측값 : ', temp[:5])
# print('원래값 : ',y_test[:5])

# #!############################################################
