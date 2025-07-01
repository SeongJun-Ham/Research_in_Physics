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
start_time = time.time()

# hyper parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCH_SIZE = 100
DROP_OUT = 0.3
LABEL_SMOOTHING = 0.05
WEIGHT = [38061, 76716, 2754, 3240]

# weight init
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

# Gradient Cam
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Hook 등록
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        loss = output[:, class_idx]
        loss.backward()

        # Global average pooling
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # shape: [B, C, 1, 1]
        grad_cam_map = (weights * self.activations).sum(dim=1, keepdim=True)
        grad_cam_map = F.relu(grad_cam_map)

        # Normalize to [0,1]
        grad_cam_map -= grad_cam_map.min()
        grad_cam_map /= (grad_cam_map.max() + 1e-8)

        return grad_cam_map
    
def apply_gradcam_to_image(model, image_tensor, class_idx=None):
    # Grad-CAM 객체 생성 (self.layer3의 마지막 conv 사용)
    cam = GradCAM(model, model.layer3[0])  # Conv2d(64, 128, kernel_size=2)

    # 배치 차원 추가 및 device로 이동
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # Grad-CAM 맵 생성
    cam_map = cam.generate(input_tensor, class_idx=class_idx)
    cam_map = cam_map.squeeze().cpu().numpy()

    # 원본 이미지
    input_image = image_tensor.squeeze().cpu().numpy()

    # 시각화용
    cam_map_resized = cv2.resize(cam_map, (input_image.shape[1], input_image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # 흑백 원본 이미지 -> 3채널로 변환
    input_image_rgb = np.stack([input_image]*3, axis=2)
    input_image_rgb = input_image_rgb / input_image_rgb.max()

    superimposed = heatmap * 0.5 + input_image_rgb * 0.5
    plt.imshow(superimposed)
    plt.title(f'Grad-CAM (class {class_idx})')
    plt.axis('off')
    plt.show()

# main
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
            torch.nn.Linear(256, 4)
        )

    # forward
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    print("call the main function")
    # device GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # directory path info
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(parent_dir, "new_DL_data")
    train_dir = os.path.join(base_dir, "training")
    validation_dir = os.path.join(base_dir, "validation")
    test_dir = os.path.join(base_dir, "test")

    # dataset transform info
    trans_info = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # data set
    training_set = torchvision.datasets.ImageFolder(
        root = train_dir,
        transform = trans_info
    )
    validation_set = torchvision.datasets.ImageFolder(
        root = validation_dir,
        transform = trans_info
    )
    # test_set = torchvision.datasets.ImageFolder(
    #     root = test_dir,
    #     transform = trans_info
    # )

    weights = 1.0 / torch.tensor(WEIGHT, dtype=torch.float)
    samples_weight = [weights[label] for _, label in training_set]
    WeightedSampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    # data loader
    training_loader = torch.utils.data.DataLoader(
        dataset = training_set,
        batch_size = BATCH_SIZE,
        sampler=WeightedSampler
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset = validation_set,
        batch_size = BATCH_SIZE
    )
    # test_loader = torch.utils.data.DataLoader(
    #     dataset = test_set,
    #     batch_size = BATCH_SIZE
    # )

        
    model = CNN().to(device)
    model.apply(init_weights)

    # Loss and Optimizer

    ## FocalLoss + log_weight
    # loss_weight = torch.tensor(WEIGHT, dtype=torch.float32)
    # log_weights = torch.log(loss_weight.max() / loss_weight)
    # weights = log_weights / log_weights.sum() * len(WEIGHT)
    # criterion = FocalLoss.FocalLoss(gamma=2.0, weight=None)

    ## CrossEntropyLoss + inverse_weight 
    inv_weights = 1.0 / torch.tensor(WEIGHT, dtype=torch.float32)
    norm_weights = inv_weights / inv_weights.sum() * len(WEIGHT)
    criterion = torch.nn.CrossEntropyLoss(weight=norm_weights.to(device), label_smoothing=0.05)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training and Validation
    training_loss = []
    validation_loss = []
    validation_accuracy = []
    for epoch in range(EPOCH_SIZE):
        print(f"epoch: {epoch+1} / {EPOCH_SIZE}")
        model.train()

        # training
        label_counts = [0]*4
        for i, (images, labels) in enumerate(training_loader):
            optimizer.zero_grad()
            for label in labels:
                label_counts[label.item()] += 1
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

            print(f"per epoch: {i} / {len(training_loader)}", end='\r')
        print(f"label_counts: {label_counts}")
        print(f"training loss: {loss.item()}")
        training_loss.append(loss.item())

        # validation
        model.eval()
        with torch.no_grad():
            flag = 1
            correct = 0
            total = 0
            class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

            for i, (images, labels) in enumerate(validation_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                if flag < 5:
                    print(f"softmax output of first image in batch: {probs[0]}")
                    flag += 1

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for pred in predicted.cpu().numpy():
                    class_counts[pred] += 1
                print(f"validation: {i} / {len(validation_loader)}", end='\r')

            validation_loss.append(loss.item())

            validation_accuracy.append(100*correct / total)
            print(f"validation dist: {class_counts}")
            print(f"validation loss: {loss.item()}")
            print("-"*100)

    # print the result
    print(f"time: {time.time() - start_time}")
    X_label = [i for i in range(EPOCH_SIZE)]
    
    plt.subplot(1,3,1)
    plt.plot(np.array(X_label), np.array(training_loss), color = 'red', label = 'train_loss')
    plt.legend(loc='lower center')

    plt.subplot(1,3,2)
    plt.plot(np.array(X_label), np.array(validation_loss), color = 'green', label = 'validation_loss')
    plt.legend(loc='lower center')

    plt.subplot(1,3,3)
    plt.plot(np.array(X_label), np.array(validation_accuracy), color = 'blue', label = 'validation_accuracy')
    plt.legend(loc='lower center')

    plt.show()

"""
sample_img, sample_label = validation_set[0]
apply_gradcam_to_image(model, sample_img, class_idx=sample_label)
"""
