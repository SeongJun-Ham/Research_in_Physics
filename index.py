import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 데이터 경로 설정
base_dir = os.getcwd() + "\Research_in_Physics"
train_data_dir = os.path.join(base_dir)
cat_data_dir = os.path.join(train_data_dir, 'cat_picture')
dog_data_dir = os.path.join(train_data_dir, 'dog_picture')

# 이미지 전처리 및 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# train_generator를 사용하여 cat과 dog 이미지를 가져옵니다.
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),  # 이미지 크기
    batch_size=32,
    class_mode='binary',  # 이진 분류
    classes=['cat_picture', 'dog_picture']  # 클래스 이름 설정
)

# CNN 모델 정의
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # 이진 분류를 위한 sigmoid 활성화 함수

# 모델 컴파일
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 모델 학습
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10  # 원하는 에폭 수로 조정
)


# 모델 저장
model.save('dog_cat_classifier.h5')
