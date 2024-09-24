import pandas as pd
import torch
import autograd.numpy as np
from autograd import jacobian
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

train_set = pd.read_csv('/Users/arasvalizadeh/Desktop/Training_Set.csv')
test_set = pd.read_csv('/Users/arasvalizadeh/Desktop/Test_Set.csv')
full_set = pd.read_csv('/Users/arasvalizadeh/Desktop/Training_Set.csv')

test_set = test_set.sample(n=500, random_state=42).reset_index(drop=True)
train_set_number = 500

train_set_features = train_set.iloc[:, 1:-1].values
full_set_features_targets = full_set.iloc[:, 1::].values 
full_set_targets = full_set.iloc[:,-1].values

high_target = train_set[train_set.iloc[:, -1] > 0.01]
low_target = train_set[train_set.iloc[:, -1] <= 0.01]
high_target_count = int(train_set_number * 0.65) 
low_target_count = train_set_number - high_target_count 
high_target_sample = high_target.sample(n=high_target_count, replace=True)
low_target_sample = low_target.sample(n=low_target_count, replace=False)

train_set = pd.concat([high_target_sample, low_target_sample]).reset_index(drop=True)
train_set = shuffle(train_set)

X_train = train_set.iloc[:, 1:-1].values
Y_train = train_set.iloc[:, -1].values
X_test = test_set.iloc[:, 1:-1].values
Y_test = test_set.iloc[:, -1].values

# # scaler = StandardScaler()

# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)
# # full_set_features = scaler.fit_transform(full_set_features)

X_train = torch.from_numpy(X_train).type(torch.float32)
Y_train = torch.from_numpy(Y_train).type(torch.float32)
X_test = torch.from_numpy(X_test).type(torch.float32)
Y_test = torch.from_numpy(Y_test).type(torch.float32)

def gaussian_membership(x, mean, L):
    transformed_x = torch.matmul(L, x)
    transformed_mean = torch.matmul(L, mean)
    diff = transformed_x - transformed_mean
    exponent = -1 * torch.matmul(diff, diff)
    return torch.exp(exponent)

def optimal_kmeans(X , Y):
    max_clusters = 10
    best_n_clusters = 1
    best_silhouette = -1
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_n_clusters = n_clusters

    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42).fit(X)
    cluster_centers = kmeans.cluster_centers_
    target_values = cluster_centers[:,-1]
    return best_n_clusters , target_values

def initialize_means_covariances(X, n_clusters):
    # X = X[:,0:-1]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    means_array = kmeans.cluster_centers_[:, 0:-1]
    means = torch.tensor(means_array, dtype=torch.float32, requires_grad=True)
    lower_indices = torch.tril_indices(X[:,0:-1].shape[1], X[:,0:-1].shape[1])
    L_componenets = []
    for i in range(n_clusters):
        cluster_points = X[:,0:-1][kmeans.labels_ == i]
        if cluster_points.shape[0] > 1:
            cov_matrix = torch.linalg.inv(torch.from_numpy(np.cov(cluster_points.T))) 
            l = torch.linalg.cholesky(cov_matrix)
        else:
            cov_matrix = torch.eye(X.shape[1]) * 1e-6
            l = torch.tensor(cov_matrix, dtype=torch.float32)
        l_componenets = l[lower_indices[0], lower_indices[1]]
        L_componenets.append(l_componenets)
    L_elements = torch.stack(L_componenets)
    L_elements = L_elements.type(torch.float32)
    L_elements.type(torch.float32).requires_grad_(True)
    return means , L_elements, lower_indices

def reconstruct_L(L_elements, lower_indices, n_clusters):
    L_matrices = []
    for i in range(n_clusters):
        L = torch.zeros((lower_indices[0].max() + 1, lower_indices[1].max() + 1), dtype=torch.float32)
        L[lower_indices[0], lower_indices[1]] = L_elements[i]
        L_matrices.append(L)
    return torch.stack(L_matrices)

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
    Y_pred = fuzzy_neural_network(X, means, w,reconstruct_L(L_elements , lower_indices , n_clusters))
    residuals = Y_pred - Y
    # print("residuals",residuals)
    # loss = residuals.sum()
    # loss.backward(retain_graph=True)
    # print(w.grad)
    # print(L_elements.grad)
    # print(means.grad)
    # print("total loss is:")
    # print(loss)
    mse = mean_squared_error(Y_pred, Y)
    def model_fn(params):
        w_, means_, L_elements_ = torch.split(params, [w.numel(), means.numel(), L_elements.numel()])
        w_ = w_.reshape(w.shape)
        means_ = means_.reshape(means.shape)
        L_elements_ = L_elements_.reshape(L_elements.shape)
        return fuzzy_neural_network(X, means_, w_ ,reconstruct_L(L_elements_ , lower_indices , n_clusters)) - Y
    jacobian_matrix = torch.autograd.functional.jacobian(model_fn, alpha)
    # print(jacobian_matrix.shape)
    # print(jacobian_matrix)

    '''
    # Sum over samples to get total gradients
    jacobian_sum = jacobian_matrix.sum(dim=0)

    # Split jacobian_sum to match parameter shapes
    w_size = w.numel()
    means_size = means.numel()
    L_elements_size = L_elements.numel()

    w_jacobian_sum = jacobian_sum[:w_size].reshape(w.shape)
    means_jacobian_sum = jacobian_sum[w_size:w_size+means_size].reshape(means.shape)
    L_elements_jacobian_sum = jacobian_sum[w_size+means_size:].reshape(L_elements.shape)

    # Compare with .grad
    print("\nGradients from Jacobian sum:")
    print("w_jacobian_sum:", w_jacobian_sum)
    print("means_jacobian_sum:", means_jacobian_sum)
    print("L_elements_jacobian_sum:", L_elements_jacobian_sum)

    # Compute differences
    print("\nDifferences:")
    print("Difference in w.grad:", torch.norm(w.grad - w_jacobian_sum))
    print("Difference in means.grad:", torch.norm(means.grad - means_jacobian_sum))
    print("Difference in L_elements.grad:", torch.norm(L_elements.grad - L_elements_jacobian_sum))
    '''
    residuals = residuals.reshape(-1, 1)  
    JTe = jacobian_matrix.T @ residuals
    H = jacobian_matrix.T @ jacobian_matrix
    update = torch.linalg.solve(H + Lambda * torch.eye(H.shape[0]), JTe)
    alpha = alpha - update.squeeze()
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

n_clusters , w = optimal_kmeans(full_set_features_targets , full_set_targets)
# print("n_clusters:",n_clusters)
# print("w",w)
w = torch.from_numpy(w).type(torch.float32).requires_grad_(True)
means , L_elements, lower_indices = initialize_means_covariances(full_set_features_targets , n_clusters)
# print("menas",means)
# print("L_elemenets",L_elements)
# print("L",reconstruct_L(L_elements , lower_indices , n_clusters))
# print("type of means",type(means))
# print("type of w",type(w))
# print("type of L",type(L_elements))
alpha = torch.cat([w.flatten() , means.flatten() , L_elements.flatten()])
# print("toole alpha (tedad parameter)",len(alpha))
# print(alpha)
max_iterations = 20
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