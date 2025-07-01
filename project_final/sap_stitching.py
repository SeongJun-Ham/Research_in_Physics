# import lib
import os
import torch
import torchvision
import torchvision.transforms as transforms
import FocalLoss
from torch.utils.data import WeightedRandomSampler

# test lib
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
torch.set_printoptions(profile="full")

start_time = time.time()

# hyper parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCH_SIZE = 50
DROP_OUT = 0.3
LABEL_SMOOTHING = 0
WEIGHT = [38061, 76716, 2754, 3240]

# weight init
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

# define CNN
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=2), # [55, 55, 32]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # [28, 28, 32]
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=2), # [27, 27, 64]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # [14, 14, 64]
        )
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=2), # [13, 13, 128]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # [7, 7, 128]
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(7*7*128, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=DROP_OUT),
            torch.nn.Linear(256, 4),
        )

    # forward
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

# uint16totensor
class Uint16ToTensor:
    def __call__(self, img: Image.Image):
        arr = np.array(img).astype(np.float32) / 65535.0
        return torch.from_numpy(arr).unsqueeze(0)

if __name__ == "__main__":
    print("call the main function")
    # device GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # directory path info
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(parent_dir, "new_DL_data")
    test_dir = os.path.join(base_dir, "test")
    test_images_path = [os.path.join(test_dir, f"{fname}.png") for fname in range(1, 77603)]

    # dataset transform info
    trans_info = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        Uint16ToTensor()
    ])
        
    model = CNN().to(device)
    model.apply(init_weights)

    # Loss and Optimizer
    ## CrossEntropyLoss + inverse_weight 
    inv_weights = 1.0 / torch.tensor(WEIGHT, dtype=torch.float32)
    norm_weights = inv_weights / inv_weights.sum() * len(WEIGHT)
    criterion = torch.nn.CrossEntropyLoss(weight=norm_weights.to(device), label_smoothing=0.05)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # stitching
    model.load_state_dict(torch.load("best_model_epoch50_D3_L05.pth", map_location=device, weights_only = True))
    model.eval()

    # 색상 매핑
    # 예측 후 stitching
    COLOR_MAP = {
        0: (255, 255, 255),     # 흰
        1: (0, 255, 0),     # 초록
        2: (0, 0, 255),     # 파랑
        3: (255, 0, 0),   # 빨강
    }

    output_image = np.zeros((322 * 56, 241 * 56, 3), dtype=np.uint8)
    for fname in tqdm(sorted(os.listdir(test_dir), key=lambda x: int(x.replace(".png", ""))), total=77602):
        if not fname.endswith(".png"):
            continue
        index = int(fname.replace(".png", "")) - 1
        y = index // 241
        x = index % 241

        img_path = os.path.join(test_dir, fname)
        image = Image.open(img_path)
        img_np = np.array(image)
        img_np = img_np.astype(np.float32)
        img_np /= 65535.0
        image_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            # probs = torch.nn.functional.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()

        color = COLOR_MAP[pred]
        output_image[y*56:(y+1)*56, x*56:(x+1)*56] = color

    # 저장
    stitched = Image.fromarray(output_image)
    stitched.save("test3.png")
    stitched.show()
