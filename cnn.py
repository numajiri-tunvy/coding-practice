import torch
import numpy as np
from torchvision import datasets, transforms
import shutil
import torch
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
import cv2

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("using device: ", device)

isDownload = False
isDelete = False

def predict(model, loss_fn, x, y):
    y_pred = model(x)
    _, predicted = torch.max(y_pred.data, 1)
    # 損失計算
    loss = loss_fn(y_pred, y)
    # 正解数
    correct = (predicted == y).sum().item()
    # 不正解のデータを抽出
    wrong_data = x[predicted != y]
    return loss, correct, wrong_data

def train(model, train_loader,loss_fn, optimizer):
    model.train()
    train_loss = 0
    train_correct = 0
    train_count = len(train_loader.dataset)
    for x, y in train_loader:
        loss, correct, _ = predict(model, loss_fn, x, y)
        # 勾配計算
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 正解数
        train_correct += correct
        train_loss += loss.item()*len(y)
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


def main():
    # データダウンロード
    dataset_train = datasets.MNIST(root='/tmp', train=True, download=isDownload, transform=torchvision.transforms.ToTensor())
    dataset_test = datasets.MNIST(root='/tmp', train=False, download=isDownload, transform=torchvision.transforms.ToTensor())

    # データをtrainとtestに分ける
    N_train = 5000
    N_test = 500

    #torch.tensor->numpy
    X_train_np = dataset_train.data.numpy()[:N_train]
    y_train_np = dataset_train.targets.numpy()[:N_train]
    X_test_np = dataset_test.data.numpy()[:N_test]
    y_test_np = dataset_test.targets.numpy()[:N_test]
    [h, w] = X_train_np.shape[1:3]

    #スケーリング0~255->0~1
    X_train_np = X_train_np/255
    X_test_np = X_test_np/255

    #trainデータをtrainとvalidationに分ける
    X_train_np, X_valid_np, y_train_np, y_valid_np = train_test_split(
        X_train_np, y_train_np, train_size = round(N_train*0.8), random_state=0
        )
    
    #サイズ変更: 1*h*wにreshape
    X_train_fig = X_train_np.reshape(X_train_np.shape[0], 1, h, w)
    X_valid_fig = X_valid_np.reshape(X_valid_np.shape[0], 1, h, w)
    X_test_fig = X_test_np.reshape(X_test_np.shape[0], 1, h, w)

    # パラメター初期設定
    torch.manual_seed(0)
    batch_size = 128

    #numpy -> torch.tensor
    X_train = torch.tensor(X_train_fig, dtype=torch.float).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.long).to(device)
    X_valid = torch.tensor(X_valid_fig, dtype=torch.float).to(device)
    y_valid = torch.tensor(y_valid_np, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test_fig, dtype=torch.float).to(device)
    y_test = torch.tensor(y_test_np, dtype=torch.long).to(device)

    # データをtensorデータセットに変換
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    # dataloader作成
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size, shuffle=False
    )

    # モデル定義
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, (5,5), padding=2), # input 28x28 -> 28x28
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2), # 28x28 -> 14x14
        torch.nn.Conv2d(16,32, (5,5)), # 14x14 -> 10x10
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2), #10x10->5x5
        torch.nn.Flatten(), # 5x5x32 -> 800
        torch.nn.Linear(800, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256,10)
    ).to(device)

    # モデルの訓練
    EPOCH = 50
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr = 0.01)
    best_model_state = cnn.state_dict()#モデルのパラメータを保存
    best_valid_acc = 0
    for epoch in range(EPOCH):
        train_loss, train_acc = train(cnn, train_loader, loss_fn, optimizer)
        valid_loss, valid_acc, _ = evaluate(cnn, valid_loader, loss_fn)
        print(f"Epoch {epoch+1}/{EPOCH}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
        if(valid_acc > best_valid_acc):
            best_model_state = cnn.state_dict()
            best_valid_acc = valid_acc

    # 最良のモデルを呼び出し
    cnn.load_state_dict(best_model_state)
    # モデルの評価
    test_loss, test_acc, wrong_data = evaluate(cnn, test_loader, loss_fn)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    data_w = wrong_data[np.random.randint(0,len(wrong_data)-1)].unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred = cnn(data_w)
        _, predicted = torch.max(y_pred, 1)
        print(f"Predicted: {predicted.item()}")
    # 不正解のデータを表示
    data_img = np.uint8(np.clip(data_w.cpu().numpy()[0,0,:]*255, 0, 255))
    print(data_img.shape)
    cv2.namedWindow("wrong_data", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("wrong_data", data_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



main()
if(isDelete):
    shutil.rmtree('/tmp', ignore_errors=True)
    
    