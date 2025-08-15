from torchvision import datasets, transforms
import shutil
import torch
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
import cv2

def main():
    # データダウンロード
    transform = transforms.ToTensor()
    dataset_train = datasets.MNIST(root='/tmp', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataset_test = datasets.MNIST(root='/tmp', train=False, download=True, transform=torchvision.transforms.ToTensor())
    
    # データをtrainとtestに分ける
    N_train = 1000
    N_test = 500
    X_train =dataset_train.data.numpy()[:N_train]/255
    y_train = dataset_train.targets.numpy()[:N_train]
    X_test = dataset_test.data.numpy()[:N_test]/255
    y_test = dataset_test.targets.numpy()[:N_test]
    [imgh, imgw] = X_train.shape[1:3]
    # trainデータをtrainとvalidationに分ける
    X_train, X_valid, y_train,y_valid = train_test_split(
        X_train, y_train, train_size=round(N_train*0.8), random_state=0
        )
    # データを28*28の一次元配列にする
    X_train_flatten = X_train.reshape(X_train.shape[0], imgh*imgw)
    X_valid_flatten = X_valid.reshape(X_valid.shape[0], imgh*imgw)
    X_test_flatten = X_test.reshape(X_test.shape[0], imgh*imgw)

    #データをtorch.tensorに変換
    X_train_torch = torch.tensor(X_train_flatten, dtype=torch.float)
    X_valid_torch = torch.tensor(X_valid_flatten, dtype=torch.float)
    X_test_torch = torch.tensor(X_test_flatten, dtype=torch.float)
    y_train_torch = torch.tensor(y_train, dtype=torch.long) # 0-9のラベル
    y_valid_torch = torch.tensor(y_valid, dtype=torch.long)
    y_test_torch = torch.tensor(y_test, dtype=torch.long)

    # パラメータ初期設定
    torch.manual_seed(0)

    # データをtensorデータセットに変換
    train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
    valid_dataset = torch.utils.data.TensorDataset(X_valid_torch, y_valid_torch)
    test_dataset = torch.utils.data.TensorDataset(X_test_torch, y_test_torch)

    # データローダーを作成
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    # 多層パーセプトロンモデルを構築
    model = torch.nn.Sequential(
        torch.nn.Linear(imgh*imgw, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )

    # 損失関数と最適化器を定義
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    #モデルを訓練
    EPOCH = 100
    for epoch in range(EPOCH):
        train_loss = 0
        train_correct = 0
        for x, y in train_loader:
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            # 損失を計算
            loss = loss_fn(y_pred, y)
            # 勾配を計算
            optimizer.zero_grad()
            loss.backward()
            # パラメータ更新
            optimizer.step()
            # 正解数
            train_correct += (predicted == y).sum().item()
        train_acc = train_correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCH}, Loss: {train_loss/len(train_loader.dataset)}, Accuracy: {train_acc}")
    
        # モデルを検証
        model.eval()
        valid_correct = 0
        for x, y in valid_loader:
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            valid_correct += (predicted == y).sum().item()
        valid_acc = valid_correct / len(valid_loader.dataset)
        print(f"Test Accuracy: {valid_acc}")

        model.train()

    
    # モデルを評価
    model.eval()
    sample_img = X_test_torch[0]
    sample_label = y_test_torch[0]

    with torch.no_grad():
        y_pred = model(sample_img.unsqueeze(0))
        predicted_label = torch.argmax(y_pred, dim = 1).item()
        print(torch.max(y_pred, dim = 1))
    print(f"正解ラベル: {sample_label}")
    print(f"モデルの予測結果: {predicted_label}")

    cv2.imshow("sample_img", sample_img.numpy().reshape(28, 28))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # データ削除
    shutil.rmtree('/tmp/MNIST', ignore_errors=True)

main()