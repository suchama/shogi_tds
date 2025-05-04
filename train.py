import torch
from torch.utils.data import DataLoader

# 既に作成した Dataset クラス
#test
dataset = ShogiDataset(folder_path='path/to/csa_files')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# モデル
model = ShogiCNN()

# GPU 対応（使える場合）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 損失関数と最適化
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# エポック数
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 順伝播
        outputs = model(inputs)

        # 損失計算
        loss = criterion(outputs, labels)

        # 逆伝播とパラメータ更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ロス積算
        total_loss += loss.item()

        # 精度計算
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_shogi_cnn_model.pth')
        print(f'✅ モデルを保存しました！（精度: {best_accuracy:.2f}％）')
