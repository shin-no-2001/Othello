# https://qiita.com/TokyoYoshida/ite50ms/07bd3cdca6a7e50c3114 参考URL

# TODO
# 訓練・推論の確認
# SDG⇒Adamを試してみる

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

# データの読み込みとシャッフル（対応関係を保存しながら）
# x_train:黒の盤面,白の盤面　y_train_reshape:次に指すべき一手
x_train = np.load("./train_data/train_data.npy")
y_train_reshape = np.load("./teacher_data/teacher_data.npy")
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train_reshape = y_train_reshape[indices]

# 学習データとテストデータに分割
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train_reshape, test_size=0.2, random_state=42)

# データをPyTorchのテンソルに変換
x_train_tensor = torch.tensor(x_train_split, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_split, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val_split, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_split, dtype=torch.float32)

# データセットとデータローダーの作成
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class Bias(nn.Module):
    def __init__(self, input_shape):
        super(Bias, self).__init__()
        self.W = nn.Parameter(torch.zeros(input_shape[1:]))
    
    def forward(self, x):
        return x + self.W

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.permute = lambda x: x.permute(0, 2, 3, 1)
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv13 = nn.Conv2d(128, 1, kernel_size=1, bias=False)
        self.flatten = nn.Flatten()
        self.bias = Bias((1, 64))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #x = self.permute(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.conv13(x)
        x = self.flatten(x)
        x = self.bias(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    model = Net().to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.92)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # トレーニングループの定義
    num_epochs = 25
    best_val_loss = float('inf')
    best_model_path = 'best_model.pth'

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs=inputs.to("cuda")
            labels=labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            # one-hotエンコードされたラベルからクラスインデックスに変換する
            _, labels_index = torch.max(labels, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels_index).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total_train)

        # バリデーション
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs=inputs.to("cuda")
                labels=labels.to("cuda")
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                # one-hotエンコードされたラベルからクラスインデックスに変換する
                _, labels_index = torch.max(labels, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels_index).sum().item()
        
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct_val / total_val)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f'Best model saved with validation loss: {best_val_loss}')

    # グラフの出力
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.show()