# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from PIL import Image
# import os


# base_dir = os.path.dirname(os.path.abspath(__file__))
# cats_dir = os.path.join(base_dir, "train_sets", "cats")
# dogs_dir = os.path.join(base_dir, "train_sets", "dogs")


# image_files_cats = [f for f in os.listdir(cats_dir)]

# trans = transforms.ToTensor()

# mean = torch.zeros(3)
# std = torch.zeros(3)
# total_images = len(image_files_cats)

# # 이미지 로드 및 처리
# valid_images = 0  # 유효한 이미지 수

# for img_file in image_files_cats:
#     img_path = os.path.join(cats_dir, img_file)
    
#     try:
#         image = Image.open(img_path).convert('RGB')  # 이미지 열기 및 RGB 변환
#         tensor_image = trans(image)  # (C, H, W) 변환
        
#         if torch.isnan(tensor_image).any():  # NaN 체크
#             print(f"NaN found in image {img_file}")
#             continue
        
#         mean += tensor_image.mean(dim=[1, 2])  # 채널별 평균
#         std += tensor_image.std(dim=[1, 2])    # 채널별 표준편차
#         valid_images += 1
    
#     except Exception as e:
#         print(f"Error loading image {img_file}: {e}")

# # 평균과 표준편차 계산 (유효한 이미지 개수로 나눔)
# if valid_images > 0:
#     mean /= valid_images
#     std /= valid_images
# else:
#     print("No valid images found to calculate mean and std.")

# cats_Mean = mean.tolist()
# cats_STD = std.tolist()

# # 아래부터 dogs

# image_files_dogs = [f for f in os.listdir(dogs_dir)]

# trans = transforms.ToTensor()

# mean = torch.zeros(3)
# std = torch.zeros(3)
# total_images = len(image_files_dogs)

# # 이미지 로드 및 처리
# valid_images = 0  # 유효한 이미지 수

# for img_file in image_files_dogs:
#     img_path = os.path.join(dogs_dir, img_file)
    
#     try:
#         image = Image.open(img_path).convert('RGB')  # 이미지 열기 및 RGB 변환
#         tensor_image = trans(image)  # (C, H, W) 변환
        
#         if torch.isnan(tensor_image).any():  # NaN 체크
#             print(f"NaN found in image {img_file}")
#             continue
        
#         mean += tensor_image.mean(dim=[1, 2])  # 채널별 평균
#         std += tensor_image.std(dim=[1, 2])    # 채널별 표준편차
#         valid_images += 1
    
#     except Exception as e:
#         print(f"Error loading image {img_file}: {e}")

# # 평균과 표준편차 계산 (유효한 이미지 개수로 나눔)
# if valid_images > 0:
#     mean /= valid_images
#     std /= valid_images
# else:
#     print("No valid images found to calculate mean and std.")

# dogs_Mean = mean.tolist()
# dogs_STD = std.tolist()

# print(f"cats_Mean: {cats_Mean}")
# print(f"cats_STD: {cats_STD}")
# print(f"dogs_Mean: {dogs_Mean}")
# print(f"dogs_STD: {dogs_STD}")



import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

# 데이터셋 불러오기 (예: CIFAR-10)
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

# 평균과 표준편차 계산
mean = 0.0
std = 0.0
total_images = len(dataset)

for image, _ in dataset:
    mean += image.mean([1, 2])  # 이미지의 채널에 대한 평균 계산
    std += image.std([1, 2])    # 이미지의 채널에 대한 표준편차 계산

mean /= total_images
std /= total_images

print(f'Mean: {mean}')
print(f'STD: {std}')