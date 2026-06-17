import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ACTIVATION = 'sigmoid'  # 'sigmoid' or 'relu'
OPTIMIZER = 'adam'  # 'sgd' or 'adam'

print(f"Using device: {DEVICE}")
print(f"Activation: {ACTIVATION}, Optimizer: {OPTIMIZER}")

# 数据预处理：转换为张量并标准化到 [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 定义简单 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self, activation='relu'):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

        # 根据参数选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("activation must be 'relu' or 'sigmoid'")

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平图像
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleMLP(activation=ACTIVATION).to(DEVICE)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 根据参数选择优化器
if OPTIMIZER == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    raise ValueError("optimizer must be 'sgd' or 'adam'")

# 用于记录训练过程
train_losses = []
train_accs = []
test_accs = []

# 训练一个 epoch
def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 评估测试集
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 主训练循环
if __name__ == '__main__':
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch()
        test_acc = evaluate()

        # 记录数值
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1}/{EPOCHS} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Test Acc: {test_acc:.2f}%')

    print("Training finished.")

    # 绘制损失曲线和准确率曲线
    plt.figure(figsize=(12, 5))

    # 子图1：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()

    # 子图2：准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accs, marker='s', label='Train Acc')
    plt.plot(range(1, EPOCHS + 1), test_accs, marker='^', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')  # 保存图像
    plt.show()  # 显示图像