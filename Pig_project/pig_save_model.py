import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. data 불러오기 

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    'D:\\pig_project\\_data\\Training',    
        target_size=(150,150), 
        batch_size=10000,
        class_mode='binary',    
        shuffle=True)


xy_test = test_datagen.flow_from_directory(
        'D:\\pig_project\\_data\\Validation',
        target_size=(150,150),
        batch_size=1000,
        class_mode='binary'
)

# print(xy_train[0][0].shape, xy_train[0][1].shape)
# print(xy_test[0][0].shape, xy_test[0][1].shape)


np.save('D:\\pig_project\\npy\\_train_pig_x.npy', arr=xy_train[0][0])
np.save('D:\\pig_project\\npy\\_train_pig_y.npy', arr=xy_train[0][1])
np.save('D:\\pig_project\\npy\\_test_pig_x.npy', arr=xy_test[0][0])
np.save('D:\\pig_project\\npy\\_test_pig_y.npy', arr=xy_test[0][1])
