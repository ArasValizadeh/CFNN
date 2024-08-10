import pandas as pd
import torch
import torch.nn as nn
import numpy as np

train_set = pd.read_csv('/Users/arasvalizadeh/Desktop/Training_Set.csv')
test_set = pd.read_csv('/Users/arasvalizadeh/Desktop/Test_Set.csv')

train_set = train_set.sample(n=2, random_state=42).reset_index(drop=True)
test_set = test_set.sample(n=2, random_state=42).reset_index(drop=True)

X_train = train_set.iloc[:, 1:-1].values
Y_train = train_set.iloc[:, -1].values

X_test = test_set.iloc[:, 1:-1].values
Y_test = test_set.iloc[:, -1].values

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

def gaussian_membership(x, mean, gamma):
    diff = x - mean
    L = torch.linalg.cholesky(gamma) 
    y = torch.matmul(L, diff)
    exponent = -torch.matmul(y, y)
    return torch.exp(exponent)

def fuzzy_neural_network(X, means, gammas, w):
    outputs = []
    for x in X:
        memberships = torch.stack([
            gaussian_membership(x, means[i], gammas[i]) for i in range(len(w))
        ])
        output = torch.dot(w, memberships)
        print("output",output)
        outputs.append(output)
    return torch.stack(outputs)


def mean_squared_error(Y_pred, Y):
    return torch.mean((Y_pred - Y) ** 2)

def lm_update(alpha, X, Y, means, gammas, w, lambd=0.01):
    Y_pred = fuzzy_neural_network(X, means, gammas, w)
    mse = mean_squared_error(Y_pred, Y)
    means.retain_grad()
    gammas.retain_grad()
    w.retain_grad()
    mse.backward()
    with torch.no_grad():
        def model_fn(params):
            w_, means_, gammas_ = torch.split(params, [w.numel(), means.numel(), gammas.numel()])
            w_ = w_.reshape(w.shape)
            means_ = means_.reshape(means.shape)
            gammas_ = gammas_.reshape(gammas.shape)
            return fuzzy_neural_network(X, means_, gammas_, w_)

        J = torch.autograd.functional.jacobian(model_fn, alpha)
        J = J.reshape(-1, alpha.numel())
        H = J.T @ J  

        mse_grad = torch.cat([w.grad.flatten(), means.grad.flatten(), gammas.grad.flatten()])
        
        mse_grad = mse_grad.unsqueeze(1) 
        
        update = torch.linalg.solve(H + lambd * torch.eye(H.shape[0]), mse_grad)  
        
        alpha -= update.squeeze()

    return alpha

def initialize_parameters(R, n):
    means = torch.randn(R, n, requires_grad=True)
    gammas = torch.eye(n).repeat(R, 1, 1).requires_grad_()
    w = torch.randn(R, requires_grad=True)
    return means, gammas, w

R = 2 
n = X_train.shape[1]  # Number of input dimensions


means, gammas, w = initialize_parameters(R, n)
alpha = torch.cat([w.flatten(), means.flatten(), gammas.flatten()])

max_iterations = 1
for epoch in range(max_iterations):
    alpha = lm_update(alpha, X_train, Y_train, means, gammas, w)

# torch.save({'means': means, 'gammas': gammas, 'w': w}, 'fuzzy_model.pth')
# Y_pred_test = fuzzy_neural_network(X_test, means, gammas, w)
# test_mse = mean_squared_error(Y_pred_test, Y_test)
# print("Test MSE:", test_mse.item())
# print("Predicted outputs:", Y_pred_test)
# print("Actual outputs:", Y_test)