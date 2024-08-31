import pandas as pd
import torch
import autograd.numpy as np
from autograd import jacobian
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

train_set = pd.read_csv('/Users/arasvalizadeh/Desktop/Training_Set.csv')
test_set = pd.read_csv('/Users/arasvalizadeh/Desktop/Test_Set.csv')
full_set = pd.read_csv('/Users/arasvalizadeh/Desktop/Training_Set.csv')
train_set_number = 1000

train_set_features = train_set.iloc[:,1:-1].values
full_set_featuers = full_set.iloc[:,1:-1].values
# scaler = StandardScaler()
# train_set_features_scaled = scaler.fit_transform(train_set_features)

high_target = train_set[train_set.iloc[:, -1] > 0.1]
low_target = train_set[train_set.iloc[:, -1] <= 0.1]
high_target_count = int(train_set_number * 0.8) 
low_target_count = train_set_number - high_target_count 
high_target_sample = high_target.sample(n=high_target_count, replace=True)
low_target_sample = low_target.sample(n=low_target_count, replace=False)
train_set = pd.concat([high_target_sample, low_target_sample]).reset_index(drop=True)

X_train = train_set.iloc[:, 1:-1].values
Y_train = train_set.iloc[:, -1].values

# X_train = scaler.transform(X_train)

X_test = test_set.iloc[:, 1:-1].values
Y_test = test_set.iloc[:, -1].values

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

def gaussian_membership(x, mean, L):
    transformed_x = torch.matmul(L, x)
    transformed_mean = torch.matmul(L, mean)
    diff = transformed_x - transformed_mean
    exponent = -1 * torch.matmul(diff, diff)
    return torch.exp(exponent)

def optimal_kmeans(X):
    max_clusters = 10
    best_n_clusters = 1
    best_silhouette = -1
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_n_clusters = n_clusters
    return  best_n_clusters

def initialize_means_covariances(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    means = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, requires_grad=True)
    L_elements = []  
    lower_indices = torch.tril_indices(X.shape[1], X.shape[1]) 
    covariances = []
    L = []
    for i in range(n_clusters):
        cluster_points = X[kmeans.labels_ == i]
        if cluster_points.shape[0] > 1:
            cov_matrix = torch.linalg.inv(torch.tensor(np.cov(cluster_points.T) , dtype=torch.float32)) 
            # paiin ghotr bayad avaz beshe zate paiin mosalasi
            l = torch.linalg.cholesky(cov_matrix.clone().detach().float()).requires_grad_(True)
        else:
            cov_matrix = torch.eye(X.shape[1]) * 1e-6
            l = torch.tensor(cov_matrix, dtype=torch.float32).requires_grad_(True)
        L.append(l.clone().detach().float())
        # print("covariance of,",i ,"rule",np.cov(cluster_points.T))
        l_elements = l[lower_indices[0], lower_indices[1]].requires_grad_(True)
        # print("l eleemnts",l_elements)
        L_elements.append(l_elements)
    L = torch.stack(L)
    L_elements = torch.stack(L_elements)
    return means , L_elements, lower_indices

def reconstruct_L(L_elements, lower_indices, n_clusters):
    L_matrices = []
    for i in range(n_clusters):
        L = torch.zeros((lower_indices[0].max() + 1, lower_indices[1].max() + 1), dtype=torch.float32)
        L[lower_indices[0], lower_indices[1]] = L_elements[i]
        L_matrices.append(L)
    return torch.stack(L_matrices)

def initialize_w(Y, n_clusters):
    min_y, max_y = Y.min(), Y.max()
    w = (torch.rand(n_clusters) * (max_y - min_y) + min_y).requires_grad_()
    return w

def fuzzy_neural_network(X, means, w ,L):
    outputs = []
    for x in X:
        memberships = torch.stack([
            gaussian_membership(x, means[i], L[i]) for i in range(len(w))
        ])
        output = torch.dot(w,memberships)
        outputs.append(output)
    return torch.stack(outputs)

def mean_squared_error(Y_pred, Y):
    return torch.mean((Y_pred - Y) ** 2)

def lm_update(alpha, X, Y, means, w, Lambda ,L_elements , lower_indices , n_clusters ):
    means.requires_grad_()
    w.requires_grad_()
    L_elements.requires_grad_()
    Y_pred = fuzzy_neural_network(X, means, w,reconstruct_L(L_elements , lower_indices , n_clusters))
    residuals = Y_pred - Y
    mse = mean_squared_error(Y_pred, Y)
    def model_fn(params):
        w_, means_, L_elements_ = torch.split(params, [w.numel(), means.numel(), L_elements.numel()])
        w_ = w_.reshape(w.shape)
        means_ = means_.reshape(means.shape)
        L_elements_ = L_elements_.reshape(L_elements.shape)
        return fuzzy_neural_network(X, means_, w_ ,reconstruct_L(L_elements_ , lower_indices , n_clusters))
    jacobian_matrix = torch.autograd.functional.jacobian(model_fn, alpha)
    print("Jacobian Matrix:", jacobian_matrix)

    residuals = residuals.reshape(-1, 1)  
    JTe = jacobian_matrix.T @ residuals
    H = jacobian_matrix.T @ jacobian_matrix
    update = torch.linalg.solve(H + Lambda * torch.eye(H.shape[0]), JTe)
    print("alpha",alpha)
    print("update",update)
    alpha -= update.squeeze()
    print("updated aplha",alpha)
    
    w_, means_, L_elements_ = torch.split(alpha, [w.numel(), means.numel(), L_elements.numel()])
    w_ = w_.reshape(w.shape)
    means_ = means_.reshape(means.shape)
    L_elements_ = L_elements_.reshape(L_elements.shape)
    Y_pred_updated = fuzzy_neural_network(X, means_, w_,reconstruct_L(L_elements_ , lower_indices , n_clusters) )
    
    mse_updated = mean_squared_error(Y_pred_updated, Y)
    if mse_updated < mse:
        Lambda /= 10
    elif mse_updated > mse:
        Lambda *= 10

    return alpha, Lambda , w_ , means_ , L_elements_

n_clusters = optimal_kmeans(full_set)
print("n_clusters:",n_clusters)
w = initialize_w(Y_train, n_clusters)
means , L_elements, lower_indices = initialize_means_covariances(full_set_featuers , n_clusters)
print("menas",means)
print("L_elemenets",L_elements)
print("L",reconstruct_L(L_elements , lower_indices , n_clusters))
alpha = torch.cat([w.flatten() , means.flatten() , L_elements.flatten()])
print("toole alpha",len(alpha))
print(alpha)
max_iterations = 50
Lambda = 1
for epoch in range(max_iterations):
    print("in",epoch+1,"epoch")
    alpha , Lambda , w , means , L_elements = lm_update(alpha, X_train, Y_train, means, w , Lambda , L_elements , lower_indices , n_clusters)
    Y_pred_test = fuzzy_neural_network(X_train, means, w ,reconstruct_L(L_elements , lower_indices , n_clusters))
    print("mse",mean_squared_error(Y_pred_test,Y_train))
    if mean_squared_error(Y_pred_test,Y_train) < 0.01:
        break
# torch.save({'means': means,'L':L ,'w': w}, 'fuzzy_model1.pth')
# Y_pred_test = fuzzy_neural_network(X_test, means, w , L)
# test_mse = mean_squared_error(Y_pred_test, Y_test)
# print("Test MSE:", test_mse.item())
# print("Predicted outputs:", Y_pred_test)
# print("Actual outputs:", Y_test)