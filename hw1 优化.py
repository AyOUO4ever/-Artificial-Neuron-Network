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
WEIGHT_DECAY = 1e-4  # 权重衰减（L2正则化）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ACTIVATION = 'relu'  # 'sigmoid' or 'relu'
OPTIMIZER = 'sgd'  # 'adam' or 'sgd

print(f"Using device: {DEVICE}")
print(f"Activation: {ACTIVATION}, Optimizer: {OPTIMIZER}")

# 数据预处理：仅标准化（未添加数据增强，保持原样）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 改进的CNN模型，添加 Dropout
class SimpleCNN(nn.Module):
    def __init__(self, activation='relu'):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层，在第一个全连接后添加 Dropout
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)  # 50% 丢弃率
        self.fc2 = nn.Linear(128, 10)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("activation must be 'relu' or 'sigmoid'")

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)  # 添加 Dropout
        x = self.fc2(x)
        return x

model = SimpleCNN(activation=ACTIVATION).to(DEVICE)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器（添加 weight_decay）
if OPTIMIZER == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    raise ValueError("optimizer must be 'sgd' or 'adam'")

# 记录训练过程
train_losses = []
train_accs = []
test_accs = []

# 早停参数
best_test_acc = 0.0
patience = 5  # 容忍连续5轮无提升
patience_counter = 0

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

# 训练循环（加入早停）
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    test_acc = evaluate()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f'Epoch {epoch + 1}/{EPOCHS} | '
          f'Train Loss: {train_loss:.4f} | '
          f'Train Acc: {train_acc:.2f}% | '
          f'Test Acc: {test_acc:.2f}%')

    # 早停逻辑
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_fashion_cnn.pth')  # 保存最佳模型
        print(f'  -> New best model saved (Acc: {best_test_acc:.2f}%)')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

print(f"\nTraining finished. Best test accuracy: {best_test_acc:.2f}%")

# 绘制曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accs) + 1), train_accs, marker='s', label='Train Acc')
plt.plot(range(1, len(test_accs) + 1), test_accs, marker='^', label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()