import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import ShogiDataset
from architecture import ShogiCNN
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folderpath = os.path.join(BASE_DIR, "data", "wdoor2017", "2017")

dataset = ShogiDataset(folder_path=folderpath)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# モデル
model = ShogiCNN()

# GPU 対応（使える場合）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")
model = model.to(device)

# 損失関数と最適化
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# エポック数
num_epochs = 2

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

train_losses = []
accuracies = []

for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 逆伝播とパラメータ更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 精度計算
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100

    train_losses.append(avg_loss)
    accuracies.append(accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), os.path.join(save_dir, f'best_shogi_first_model.pth'))
        print(f'✅ モデルを保存しました！（精度: {best_accuracy:.2f}％）')

# 学習後にグラフで可視化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()