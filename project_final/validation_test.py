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

    # stitching
    model.load_state_dict(torch.load("best_model_epoch50_D3.pth", map_location=device, weights_only = True))
    model.eval()

    with torch.no_grad():
        flag = 1
        correct = 0
        total = 0
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        correct_per_class = {0: 0, 1: 0, 2: 0, 3: 0}
        total_per_class = {0: 0, 1: 0, 2: 0, 3: 0}
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

            a_preds = predicted.cpu().numpy()
            a_labels = labels.cpu().numpy()
            for n in range(len(a_preds)):
                label = int(a_labels[n])  # int()로 캐스팅하면 np.int64 문제 없음
                pred = int(a_preds[n])

                total_per_class[label] += 1
                if pred == label:
                    correct_per_class[label] += 1
        print(f"correct_per_class: {correct_per_class}")
        print(f"total per class: {total_per_class}")

        print(f"validation dist: {class_counts}")
        print(f"validation loss: {loss.item()}")
        print("-"*100)