import cv2
import torch
import numpy as np
from torchvision import datasets, transforms
import shutil
from sklearn.model_selection import train_test_split

#GPUが利用可能か確認
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("using device: ", device)

def predict(model, loss_fn, x, y):
    # 予測計算
    y_pred = model(x)#入力行列xの予測結果行列[N, 10]
    v, idx = torch.max(y_pred.detach(), 1)# 最大値スコア（確率）とそのインデックス（今回は0~9のラベルに相当）
    correct = (idx == y).sum().item()# 正解数
    # 不正解のデータを抽出
    wrong_data = x[idx != y]
    #損失計算
    loss = loss_fn(y_pred, y)
    return loss, correct, wrong_data


def train(model, train_loader, loss_fn, optimizer):
    model.train()#モデルを訓練モードに
    train_loss = 0
    train_correct  = 0#モデルの正解数
    train_count = len(train_loader.dataset)
    for x, y in train_loader:
        loss, correct, _ = predict(model, loss_fn, x, y)
        train_correct += correct
        train_loss += loss.item()*len(y)
        # 勾配を計算
        optimizer.zero_grad()#勾配りセット
        loss.backward()#勾配計算
        optimizer.step()#パラメータ更新
    return train_loss/train_count, train_correct/train_count

def evaluate(model, valid_loader, loss_fn):
    model.eval()
    valid_loss = 0
    valid_correct = 0
    valid_count = len(valid_loader.dataset)
    for x, y in valid_loader:
        loss, correct, wrong_data = predict(model, loss_fn, x, y)
        valid_loss += loss.item()*len(y)
        valid_correct += correct
    return valid_loss/valid_count, valid_correct/valid_count, wrong_data

def delete_data(dataDelete):
    if(dataDelete):
        shutil.rmtree("/tmp/MNIST", ignore_errors=True)
    
def main():
    dataDownload = True
    dataDelete = False
    # データダウンロード
    dataset_train = datasets.MNIST(root="/tmp", train=True, download=dataDownload, transform=transforms.ToTensor())
    dataset_test = datasets.MNIST(root="/tmp", train=False, download=dataDownload, transform=transforms.ToTensor())
    #データをnummpyに変換して正規化
    N_train = 30000
    N_test = 10000
    X_train = dataset_train.data.numpy()[:N_train]/255
    y_train = dataset_train.targets.numpy()[:N_train]
    X_test = dataset_test.data.numpy()[:N_test]/255
    y_test = dataset_test.targets.numpy()[:N_test]
    # データを訓練データと検証データに分割
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size = round(N_train*0.8), random_state=0)
    # データの画像サイズを取得
    [height, width] = X_train.shape[1:3]
    #データの画像部分を一次元に変換
    X_train_flatten = X_train.reshape(X_train.shape[0], height*width)
    X_test_flatten = X_test.reshape(X_test.shape[0], height*width)
    X_valid_flatten = X_valid.reshape(X_valid.shape[0], height*width)
    # Tensorデータに変換
    X_train_torch = torch.tensor(X_train_flatten, dtype=torch.float).to(device)
    X_valid_torch = torch.tensor(X_valid_flatten, dtype=torch.float).to(device)
    X_test_torch = torch.tensor(X_test_flatten, dtype=torch.float).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.long).to(device)
    y_valid_torch = torch.tensor(y_valid, dtype=torch.long).to(device)
    y_test_torch = torch.tensor(y_test, dtype=torch.long).to(device)
    # パラメータ初期設定
    torch.manual_seed(0)
    #データセットの再作成
    train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
    valid_dataset = torch.utils.data.TensorDataset(X_valid_torch, y_valid_torch)
    test_dataset = torch.utils.data.TensorDataset(X_test_torch, y_test_torch)
    # データローダーを作成
    batch_size = 512
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)
    #モデル作成
    model = torch.nn.Sequential(
        torch.nn.Linear(height*width, 512),
        torch.nn.PReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.Dropout(0.2),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.PReLU(),
        torch.nn.Linear(64, 10) 
    ).to(device)
    # 損失関数定義
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    # optimizerを定義 SGDとAdamの違いは？
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #モデルを訓練
    best_model_state = model.state_dict()
    best_valid_acc = 0
    EPOCH = 256
    for epoch in range(EPOCH):
        #訓練データを利用して訓練
        train_loss, train_acc = train(
            model, train_loader, loss_fn, optimizer
        )
        # 検証データを用いてモデルを評価
        valid_loss, valid_acc, _ = evaluate(
            model, valid_loader, loss_fn
        )
        # 検証データの精度が最も高いモデルを最適モデルとして記録
        if(valid_acc > best_valid_acc):
            best_model_state = model.state_dict()
            best_valid_acc = valid_acc
    # 最適モデルを読み込み
    model.load_state_dict(best_model_state)
    print(f"Best Valid Acc: {best_valid_acc}")
    # モデルを評価
    test_loss, test_acc, wrong_data = evaluate(
        model, test_loader, loss_fn
    )
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}")
    # 不正解のデータを表示
    random_index = np.random.randint(0, len(wrong_data))
    wrong_data_x = wrong_data[random_index].unsqueeze(0).to(device)#tensor
    with torch.no_grad():
        y_pred = model(wrong_data_x)
        _, predicted = torch.max(y_pred, 1)
        print(f"Predicted: {predicted.item()}")
    

    cv2.namedWindow("wrong_data", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("wrong_data", wrong_data_x.cpu().numpy().reshape(28, 28))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #データの削除
    if(dataDelete):
        shutil.rmtree("/tmp/MNIST", ignore_errors=True)

main()
delete_data(True)
