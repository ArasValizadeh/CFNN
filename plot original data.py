import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


file_path = '/Users/arasvalizadeh/Desktop/Training_Set.csv'  
data = pd.read_csv(file_path)

x1 = data['x1']
x2 = data['x2']
y = data['y']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1, x2, y, c=y, cmap='viridis', marker='o')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')

ax.set_title('3D Scatter Plot of x1, x2, and y')

plt.show()