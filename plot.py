import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

saved_model = torch.load('fuzzy_model22.pth')
means = saved_model['means']
gammas = saved_model['gammas']
w = saved_model['w']
L = saved_model['L']
def gaussian_membership(x, mean, gamma , L):
    diff = x - mean
    transformed_x = torch.matmul(L, x)
    transformed_mean = torch.matmul(L, mean)
    diff = transformed_x - transformed_mean
    exponent = -torch.matmul(diff, diff)
    return torch.exp(exponent)

def fuzzy_neural_network(X, means, gammas, w ,L):
    outputs = []
    for x in X:
        memberships = torch.stack([
            gaussian_membership(x, means[i], gammas[i] , L[i]) for i in range(len(w))
        ])
        output = torch.dot(w, memberships)
        outputs.append(output)
    return torch.stack(outputs)


x1_values = np.linspace(-10, 10, 500)
x2_values = np.linspace(-10, 10, 500)
x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
X_plot = np.c_[x1_grid.ravel(), x2_grid.ravel()]
X_plot_tensor = torch.tensor(X_plot, dtype=torch.float32)
Y_plot = fuzzy_neural_network(X_plot_tensor, means, gammas, w , L).detach().numpy()
Y_plot = Y_plot.reshape(x1_grid.shape)

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot_surface(x1_grid, x2_grid, Y_plot, cmap='viridis')
ax2.set_title('Approximated Function fuzzy model5')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')

plt.show()