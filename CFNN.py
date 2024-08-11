import pandas as pd
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

train_set = pd.read_csv('/Users/arasvalizadeh/Desktop/Training_Set.csv')
test_set = pd.read_csv('/Users/arasvalizadeh/Desktop/Test_Set.csv')

full_data = pd.read_csv('/Users/arasvalizadeh/Desktop/Training_Set.csv')

train_set = train_set.sample(n=700, random_state=42).reset_index(drop=True)
test_set = test_set.sample(n=50, random_state=42).reset_index(drop=True)

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

def optimal_kmeans(X):
    max_clusters = 10  # Set a maximum number of clusters to evaluate
    best_n_clusters = 1
    best_silhouette = -1
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_n_clusters = n_clusters
    
    return  best_n_clusters


def initialize_means_covariances(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    means = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).requires_grad_()
    covariances = torch.eye(X.shape[1]).repeat(n_clusters, 1, 1).requires_grad_()
    return means, covariances

def initialize_w(Y, n_clusters):
    min_y, max_y = Y.min(), Y.max()
    w = (torch.rand(n_clusters) * (max_y - min_y) + min_y).requires_grad_()
    return w

def fuzzy_neural_network(X, means, gammas, w):
    outputs = []
    for x in X:
        memberships = torch.stack([
            gaussian_membership(x, means[i], gammas[i]) for i in range(len(w))
        ])
        output = torch.dot(w, memberships)
        outputs.append(output)
    return torch.stack(outputs)


def mean_squared_error(Y_pred, Y):
    return torch.mean((Y_pred - Y) ** 2)

def lm_update(alpha, X, Y, means, gammas, w, Lambda):
    Y_pred = fuzzy_neural_network(X, means, gammas, w)
    mse = mean_squared_error(Y_pred, Y)
    means.retain_grad()
    gammas.retain_grad()
    w.retain_grad()
    mse.backward()
    e = Y - Y_pred
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
        
        # update = torch.linalg.solve(H + Lambda * torch.eye(H.shape[0]), mse_grad)  
        epsilon = 1e-4  # Small value to add to the diagonal
        update = torch.linalg.solve(H + (Lambda + epsilon) * torch.eye(H.shape[0]), mse_grad)
        
        alpha -= update.squeeze()
        # J = torch.autograd.functional.jacobian(model_fn, alpha)
        # J = J.reshape(-1, alpha.numel())
        # H = J.T @ J  
        # JTe = J.T @ e.unsqueeze(1) 
        # update = torch.linalg.solve(H + Lambda * torch.eye(H.shape[0]), JTe)  
        # alpha -= update.squeeze()
    
    w_, means_, gammas_ = torch.split(alpha, [w.numel(), means.numel(), gammas.numel()])
    w_ = w_.reshape(w.shape)
    means_ = means_.reshape(means.shape)
    gammas_ = gammas_.reshape(gammas.shape)

    Y_pred_updated = fuzzy_neural_network(X, means_, gammas_, w_)
    mse_updated = mean_squared_error(Y_pred_updated, Y)
    
    if mse_updated < mse:
        Lambda /= 10  
    elif mse_updated > mse:
        Lambda *= 10  

    return alpha , Lambda

n_clusters = optimal_kmeans(full_data)
print(n_clusters)
print("...........")
w = initialize_w(Y_train, n_clusters)
means , gammas = initialize_means_covariances(X_train , n_clusters)
print(w)
print("...........")
print(means)
print("...........")
print(gam)
print("...........")
R = n_clusters
n = X_train.shape[1]  # Number of input dimensions

alpha = torch.cat([w.flatten(), means.flatten(), gammas.flatten()])

# max_iterations = 8
# i = 1
# Lambda = 0.01
# for epoch in range(max_iterations):
#     print(i)
#     i+=1
#     alpha , Lambda = lm_update(alpha, X_train, Y_train, means, gammas, w , Lambda)

# torch.save({'means': means, 'gammas': gammas, 'w': w}, 'fuzzy_model2.pth')
# Y_pred_test = fuzzy_neural_network(X_test, means, gammas, w)
# test_mse = mean_squared_error(Y_pred_test, Y_test)
# print("Test MSE:", test_mse.item())
# print("Predicted outputs:", Y_pred_test)
# print("Actual outputs:", Y_test)