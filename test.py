import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 저장된 모델 로드
model = load_model('dog_cat_classifier.h5')

# 이미지 경로 설정 (분류할 이미지)
img_path = 'E:/Users/sj879/Desktop/dogcat.jpg'  # 예측할 이미지 경로로 변경하세요

# 이미지를 모델 입력 형태로 전처리
img = image.load_img(img_path, target_size=(150, 150))  # 모델에 맞는 크기로 조정
img_array = image.img_to_array(img)  # 이미지를 배열로 변환
img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
img_array /= 255.0  # 정규화

# 예측
predictions = model.predict(img_array)
print(f"{predictions}")
print(f"{type(predictions)}")
print(predictions[0][0])
if predictions[0][0] > 0.5:
    print("강아지입니다.")
else:
    print("고양이입니다.")