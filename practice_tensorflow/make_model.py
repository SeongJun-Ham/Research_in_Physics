import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# training 데이터 경로 <- 크롤링으로 가져온 사진
train_data_dir = os.path.join(os.getcwd(), "\Research_in_Physics")
cat_data_dir = os.path.join(train_data_dir, "cat_picture")
dog_data_dir = os.path.join(train_data_dir, "dog_picture")

# 데이터 전처리 세팅
""" ImageDataGenerator로 사진 전처리할 때 조정할 defalt parameters
train_data_gen = ImageDataGenerator(
    rescale = None,
    rotation_range = 0,
    width_shift_range = 0,
    height_shift_range = 0,
    shear_range = 0,
    zoom_range = 0,
    horizontal_flip = True,
    fill_mode = "constant"
)
"""
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant'
)

# 데이터 전처리
""" flow_from_directory로 사진 데이터 가져올 때 defalt parameters
train_generator = train_datagen.flow_from_directory(
    directory,
    target_size = (256, 256),
    batch_size = 32,
    class_mode = str으로 categorical, binary, sparse, input, None <- defalt는 categorical
    shuffle = True,
    seed = None,
    save_to_dir = None,
    save_prefix = '접두사',
    save_format = 'png',
    subset = None
    interpolation = str으로 nearest, bilinear, bicubic, lanczos, box <- defalt는 nearest
    classes = None
)
"""

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (100, 100),
    batch_size = 20,
    class_mode = 'binary',
    classes = ['cat_picture', 'dog_picture']
)

# CNN 모델 정의

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3)))