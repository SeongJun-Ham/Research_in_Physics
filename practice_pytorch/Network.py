import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn

# hyper parameters
BATCH_SIZE = 100
LEARNING_RATE = 0.001 # 너무 크면 overshooting, 최적점에서 벗어남, 보통 10e-n -> 바꿔서 해볼 것
EPOCH_SIZE = 10 # 전체 학습 횟수, 너무 크면 overfitting -> 바꿔서 해볼 것

# device GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# directory path info
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, "train_sets")
test_dir = os.path.join(base_dir, "test_sets")

# dataset transform info
trans_info = transforms.Compose([
    transforms.RandomAffine(15, translate=(0.1, 0.1)),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

# train and test dataset
train_set = torchvision.datasets.ImageFolder(
    root = train_dir, 
	transform = trans_info
    )

test_set = torchvision.datasets.ImageFolder(
    root = test_dir,
    transform = trans_info
)

# train and test loader
train_loader = torch.utils.data.DataLoader(
    dataset = train_set,
    batch_size = BATCH_SIZE,
    shuffle = True
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_set,
    batch_size = BATCH_SIZE
)

# define CNN
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.7 # 뉴런 생존 확률(overfitting 방지) -> 변화시켜서 해볼 것
        # [256x256x3]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1), # [256, 256, 16]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2) # [128, 128, 16]
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1), # [128, 128, 32]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2) # [64, 64, 32]
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1), # [64, 64, 64]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2) # [32, 32, 64]
        )

        # self.fc = torch.nn.Linear(32*32*64, 2)

        self.fc1 = torch.nn.Linear(32*32*64, 100)

        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-self.keep_prob)
        )

        self.fc2 = torch.nn.Linear(100, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.layer4(x)
        x = self.fc2(x)
        return x

model = CNN().to(device)

# Loss and Optimizer function
criterion = torch.nn.CrossEntropyLoss() # loss function으로 CrossEntropy -> 바꿔서 해볼 것
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train model
for epoch in range(EPOCH_SIZE):
    for i, (images, labels) in enumerate(train_loader): # len(train_loader = 16), 1600장, batchsize = 100이므로 & 각 data는 image tensor와 정답 label tensor을 가짐
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad() # 한 번 학습시킨 다음에 optimizer를 초기화해줘야 다음 학습 때 간섭 안 함
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCH_SIZE}], Loss: {loss.item():0.4f}") # Epoch 진행도 표시


# Test the model
pred = []
model.eval()  # eval mode -> Pytorch에서 모델을 평가모드로 전환하는 메서드
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        pred.append((predicted, labels))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


for p in pred:
    print(p)

print('Accuracy: {} %'.format(100 * correct / total))