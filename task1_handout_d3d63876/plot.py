import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from sklearn.cluster import KMeans

# Load the training dateset and test features
train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

fig = plt.figure()
ax = Axes3D(fig)

fig2 = plt.figure()
ax2 = Axes3D(fig2)

xy = np.column_stack((train_x, train_y))


data = np.column_stack((train_x, train_y))
kmeans = KMeans(n_clusters=5000).fit(data)
data_clustered = kmeans.cluster_centers_
ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2])
ax2.scatter(data_clustered[:, 0], data_clustered[:, 1], data_clustered[:, 2])
plt.show()
