# import lib
import os
import torch
import torchvision
import torchvision.transforms as transforms

# test lib
import time
import matplotlib.pyplot as plt
import numpy as np
start_time = time.time()

# hyper parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCH_SIZE = 50
DROP_OUT = 0
LABEL_SMOOTHING = 0.1
LOSS_WEIGHT = [38061, 76716, 2754, 3240]


# main
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=2), # [55, 55, 32]
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # [28, 28, 32]
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=2), # [27, 27, 64]
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # [14, 14, 64]
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(14*14*64, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=DROP_OUT),
            torch.nn.Linear(128, 4)
        )

    # forward
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
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

    # data loader
    training_loader = torch.utils.data.DataLoader(
        dataset = training_set,
        batch_size = BATCH_SIZE
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
    print("flag")
    print(len(validation_loader))

    # Loss and Optimizer
    WEIGHT = torch.tensor(LOSS_WEIGHT, dtype=torch.float32)
    weights = 1.0 / torch.sqrt(WEIGHT)
    WEIGHT = weights / weights.sum()
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing = LABEL_SMOOTHING,
        weight = WEIGHT.to(device)
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training and Validation
    training_loss = []
    validation_accuracy = []
    for epoch in range(EPOCH_SIZE):
        print(f"epoch: {epoch+1} / {EPOCH_SIZE}")
        model.train()

        # training
        label_counts = [0]*4
        for i, (images, labels) in enumerate(training_loader):
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
            optimizer.zero_grad()

            print(f"per epoch: {i} / {len(training_loader)}", end='\r')
        print(f"label_counts: {label_counts}")

        # validation
        model.eval()
        training_loss.append(loss.item())
        with torch.no_grad():
            correct = 0
            total = 0
            class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                print(f"softmax output of first image in batch: {probs[0]}")
                print(f"predicted class: {torch.argmax(probs[0]).item()}")

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for pred in predicted.cpu().numpy():
                    class_counts[pred] += 1

            validation_accuracy.append(100*correct / total)
            print(f"EPOCH: {epoch+1} / {EPOCH_SIZE}")
            print(f"validation dist: {class_counts}")
            print(f"loss: {loss.item()}")
            print("-"*100)

    # print the result
    print(f"time: {time.time() - start_time}")
    X_label = [i for i in range(EPOCH_SIZE)]
    
    plt.subplot(1,2,1)
    plt.plot(np.array(X_label), np.array(training_loss), color = 'red', label = 'train_loss')
    plt.legend(loc='lower center')

    plt.subplot(1,2,2)
    plt.plot(np.array(X_label), np.array(validation_accuracy), color = 'blue', label = 'validation_accuracy')
    plt.legend(loc='lower center')

    plt.show()