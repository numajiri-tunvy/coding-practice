import autograd
import autograd.numpy as np
import torch

def loss(w, x):
    return -np.log(1.0/(1+np.exp(-np.dot(x, w))))

def backpropagate(w, x):
    grad_loss = autograd.grad(loss)
    print(loss(w, x))
    print(grad_loss(w, x))

def main():
    dtype = torch.float
    grand_truth = []
    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1
        q = (a ^ b) | c
        grand_truth.append([a, b, c, q])
    N = 8
    train_index = [np.random.randint(0, 8) for _ in range(N)]
    x_train = torch.tensor([grand_truth[i][:3] for i in train_index], dtype=dtype)
    y_train = torch.tensor([[grand_truth[i][3]] for i in train_index], dtype=dtype)
    
    print(f"x_train: {x_train}")
    print(f"y_train: {y_train}")

    w1 = torch.randn(3, 3, dtype=dtype, requires_grad=True)
    w2 = torch.randn(3, 1, dtype=dtype, requires_grad=True)
    b2 = torch.randn(1, 1, dtype=dtype, requires_grad=True)
    eta = 0.5
    for t in range(100):
        y_pred = x_train.mm(w1).sigmoid().mm(w2).add(b2).sigmoid()
        ll = y_train * y_pred + (1-y_train) * (1-y_pred)
        loss = -ll.log().sum()
        print(t, loss.item())
        loss.backward()
        with torch.no_grad():
            w1 -= eta*w1.grad
            w2 -= eta*w2.grad
            b2 -= eta*b2.grad
            w1.grad.zero_()
            w2.grad.zero_()
            b2.grad.zero_()

    print(x_train.mm(w1).sigmoid().mm(w2).add(b2).sigmoid())
    print(y_train)
    # 正解率
    x_grand = torch.tensor([grand_truth[i][:3] for i in range(len(grand_truth))], dtype=dtype)
    y_grand = torch.tensor([grand_truth[i][3] for i in range(len(grand_truth))], dtype=dtype)
    y_pred_grand = x_grand.mm(w1).sigmoid().mm(w2).add(b2).sigmoid().detach().numpy()
    grand_correct = y_grand.detach().numpy()
    correct = 0
    for i in range(len(y_pred_grand)):
        v = 1 if y_pred_grand[i] > 0.5 else 0
        if v == grand_correct[i]:
            correct += 1
        else:
            print(f"grand_truth: {grand_truth[i]}")
            print(f"y_pred_grand: {y_pred_grand[i]}")
    print(f"accuracy: {correct / len(y_pred_grand)}")

main()


