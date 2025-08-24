import shutil
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# GPUの設定
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("using device: ", device)

isDownload = False
isDelete = True

def predict(model, loss_fn, x, y):
    y_pred = model(x)
    _, predicted = torch.max(y_pred, 1)
    loss = loss_fn(y_pred, y)
    # 正解数をカウント
    correct = (predicted == y).sum().item()
    # 不正解データの抽出
    wrong_data = x[predicted != y]
    return loss, correct, wrong_data

def train(model, optimizer, loss_fn, train_loader):
    model.train()
    train_loss = 0
    train_correct = 0
    train_count = len(train_loader.dataset)
    for x, y in train_loader:
        loss, correct, _ = predict(model, loss_fn, x, y)
        # 勾配を初期化
        optimizer.zero_grad()
        # 誤差逆伝播
        loss.backward()
        # パラメーター更新
        optimizer.step()
        train_loss += loss.item()*len(y)
        train_correct += correct
    return train_loss/train_count, train_correct/train_count

def evaluate(model, loss_fn, loader):
    model.eval()
    valid_loss = 0
    valid_correct = 0
    valid_count = len(loader.dataset)
    for x, y in loader:
        loss, correct, wrong_data = predict(model, loss_fn, x, y)
        valid_loss += loss.item()*len(y)
        valid_correct += correct
    return valid_loss/valid_count, valid_correct/valid_count, wrong_data


def main():
    # データダウンロード
    dataset_train_mnist = datasets.MNIST(root='tmp', train=True, download=isDownload, transform=transforms.ToTensor())
    dataset_test_mnist = datasets.MNIST(root='tmp', train=False, download=isDownload, transform=transforms.ToTensor())

    #データをtrain, testに分割
    N_train = 5000
    N_test = 500
    train_X_np = dataset_train_mnist.data.numpy()[:N_train]
    train_y_np = dataset_train_mnist.targets.numpy()[:N_train]
    test_X_np = dataset_test_mnist.data.numpy()[:N_test]
    test_y_np = dataset_test_mnist.targets.numpy()[:N_test]

    # データを0~1の範囲に正規化
    train_X_np = train_X_np/255
    test_X_np = test_X_np/255

    # データをデータ数xチャネル数x画像サイズに整形
    [h, w] = train_X_np.shape[1:3]
    train_X_fig = train_X_np.reshape(N_train, 1, h, w)
    test_X_fig = test_X_np.reshape(N_test, 1, h, w)
    
    #　データをtrain, validに分割
    train_X_fig, valid_X_fig, train_y_np, valid_y_np = train_test_split(
        train_X_fig, train_y_np, train_size = round(N_train*0.8), random_state=0
    )

    # データをTensorに変換
    train_X = torch.tensor(train_X_fig, dtype=torch.float).to(device)
    train_y = torch.tensor(train_y_np, dtype=torch.long).to(device)
    valid_X = torch.tensor(valid_X_fig, dtype=torch.float).to(device)
    valid_y = torch.tensor(valid_y_np, dtype=torch.long).to(device)
    test_X = torch.tensor(test_X_fig, dtype=torch.float).to(device)
    test_y = torch.tensor(test_y_np, dtype=torch.long).to(device)

    # データセットの作成
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    valid_dataset = torch.utils.data.TensorDataset(valid_X, valid_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)

    # dataLoaderを作成
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)

    # モデルの定義
    torch.manual_seed(0) #パラメター初期化
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, (5, 5)), # 28x28 -> 24x24
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 16, (7, 7)), #24x24 -> 18x18
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(3), #18x18 -> 6x6
        torch.nn.Flatten(), #6x6x16 -> 576
        torch.nn.Linear(576, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    ).to(device)
    optimizer = torch.optim.SGD(cnn.parameters(), lr = 0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # モデルの学習
    EPOCH = 50
    best_model_state = cnn.state_dict()
    best_valid_loss = np.inf
    for epoch in range(EPOCH):
        train_loss, train_acc = train(cnn, optimizer, loss_fn, train_loader)
        valid_loss, valid_acc, _ = evaluate(cnn, loss_fn, valid_loader)
        # 学習状況を表示
        print(f"epoch: {epoch+1}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}")
        # モデルの保存
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = cnn.state_dict()
            print("best model saved")
    
    # モデルの評価
    cnn.load_state_dict(best_model_state)
    loss, correct, wrong_data = evaluate(cnn, loss_fn, test_loader)
    print(f"test_loss: {loss:.4f}, test_acc: {correct:.4f}")
    # 間違ったデータを表示
    wrong_mps = wrong_data[np.random.randint(0, len(wrong_data)-1)].unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred=cnn(wrong_mps)
        _, predict = torch.max(y_pred, 1)
        print(f"predict: {predict.item()}")
    # 画像表示
    data_img = wrong_mps.cpu().numpy()[0,0,:,:]
    data_img = np.uint8(np.clip(data_img*255, 0, 255))
    cv2.imshow("wrong_data", data_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()

if(isDelete):
    shutil.rmtree("tmp")