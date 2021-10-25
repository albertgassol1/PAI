import numpy as np

import scipy
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel, DotProduct, ExpSineSquared, RationalQuadratic, Sum
from sklearn.kernel_approximation import Nystroem
from sklearn.cluster import KMeans

#np.random.bit_generator = np.random._bit_generator

## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04

Npoints = 450
bias = 0

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


def cost_function(true, predicted):
	"""
		true: true values in 1D numpy array
		predicted: predicted values in 1D numpy array

		return: float
	"""
	cost = (true - predicted)**2

	# true above threshold (case 1)
	mask = true > THRESHOLD
	mask_w1 = np.logical_and(predicted>=true,mask)
	mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
	mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

	cost[mask_w1] = cost[mask_w1]*W1
	cost[mask_w2] = cost[mask_w2]*W2
	cost[mask_w3] = cost[mask_w3]*W3

	# true value below threshold (case 2)
	mask = true <= THRESHOLD
	mask_w1 = np.logical_and(predicted>true,mask)
	mask_w2 = np.logical_and(predicted<=true,mask)

	cost[mask_w1] = cost[mask_w1]*W1
	cost[mask_w2] = cost[mask_w2]*W2

	reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
	if reward is None:
		reward = 0
	return np.mean(cost) - np.mean(reward)

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


	M = Model()
	M.fit_model(train_x,train_y)
	prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model():

	def __init__(self):
		
		
		self.kernel1 = ConstantKernel() + RBF() + WhiteKernel()
		self.kernel2 = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
		self.kernel3 = RationalQuadratic() + WhiteKernel() # Best performes so far
		self.kernel4 = ConstantKernel() + ConstantKernel()*RBF()
		self.kernel5 = ConstantKernel() + ConstantKernel()*RBF() + ConstantKernel()*WhiteKernel()
		self.model = GaussianProcessRegressor(kernel=self.kernel1, n_restarts_optimizer=0, random_state=0)

	def predict(self, test_x):       
		# predict with model at test point test_x 

		y = self.model.predict(test_x)
		return y

	def load_and_tranform_data(self, approx, train_x, train_y):        
		
		data_xy = np.column_stack((train_x,train_y))
		rng = np.random.default_rng()

		# truncate data set to cut off all points where x1 < -0.5
		#data_xy = data_xy[data_xy[:,0] >= -0.5, :]        

		# Nyostream approximation (not implemented yet)
		if(approx == 'Nystroem'):
			feature_map_nystroem = Nystroem(kernel='rbf',gamma=.2,random_state=1,n_components=300)
			data_transformed = feature_map_nystroem.fit_transform(train_x)

		# Select random samples from dataset
		if(approx == 'Random'):
			n = 100
			data_transformed = rng.choice(data_xy,size=n, axis=0, replace=False)            
		
		# Clusterize data into 
		if(approx == 'Clusters'):
			n_clusters = 800
			dist_thresehold = 0.07
			cluster_centers = rng.choice(data_xy,size=n_clusters, axis=0, replace=False)
			data_transformed = np.zeros((1,3))
			for i in range(n_clusters):
				cluster_center = cluster_centers[i,0:3]
				for point in data_xy:
					dist = np.linalg.norm(cluster_center-point)
					if(dist < dist_thresehold):
						point_app = np.reshape(point,(1,3))
						if(data_transformed.all()==0):
							data_transformed = point_app
							continue

						data_transformed = np.append(data_transformed,point_app,axis=0)

		# KMeans
		if(approx=='KMeans'):
			kmeans = KMeans(n_clusters=Npoints, random_state=0).fit(data_xy)
			data_transformed = kmeans.cluster_centers_


		# Using entire data set
		if(approx == 'None'):
			data_transformed = data_xy[bias:Npoints+bias,:]

		print("Size of transformed data: " + str(data_transformed.shape))    
		return data_transformed

	def fit_model(self, train_x, train_y):

		# load and transform data according to different methods
#       data = self.load_and_tranform_data('Random', train_x, train_y)
		
		# trying KMeans Method
		data_transformed = self.load_and_tranform_data('KMeans',train_x,train_y)
		self.data_x = data_transformed[:,0:2]
		self.data_y = data_transformed[:,2]

		# Tried NYOSTROEM SELECTION by concatenating data_x and data_y and using the transpose as input to Nystroem --> this didn't work well since the approximation is limited by n in an nxm matrix
		# Hence the output was only 3x3
# 		# merge data into one matrix for selection in nyostroem selector
# 		N, f = np.shape(self.data_x)

# 		data = np.zeros((N, 3))
# 		data[:,0:2] = self.data_x
# 		data[:,2] = self.data_y[:]
# 		
# 		data_trans = np.transpose(data)
# 		
# 		# TO DO: incorporate transformed data into model.fit
# 		feature_map_nystroem = Nystroem(gamma=.2, random_state=0, n_components=1000)
# 		data_transformed = feature_map_nystroem.fit_transform(data_trans)
# 		
# 		X_transformed = np.transpose(data_transformed[0:2,:])
# 		y_transformed = np.transpose(data_transformed[2,:])

		# initliaze model with kernel (necessary before actually doing optimization with our custom cost function)
		self.model.fit(self.data_x, self.data_y)                

		# actually optimize hyperparameters according to custom cost function --> Why is optimizer after fit function?
		self.model.kernel_.theta = self.optimizer()

		""" CROSS VALIDATION
		# Trying different initializartions via cross validation
		n_restarts=1
		cost_cv = np.zeros((n_restarts,1))
		theta = np.zeros((n_restarts, self.model.kernel_.theta.shape[0]))
		for i in range(n_restarts):

			# Random parametre initiallization
			random_initialization = np.random.randn(1,self.model.kernel_.theta.shape[0])[0,:]
			print('\n',random_initialization)
			self.model.kernel_.theta = random_initialization
			self.model.kernel_.theta = self.optimizer()
			print(self.model.kernel_)

			# Select new cross validation data set
			d = rng.choice(data_xy,size=200, axis=0, replace=False)
			x = d[:,0:2]
			y = d[:,2:3]
			cost_cv[i] = cost_function(y,self.model.predict(x))
			theta[i,:] = self.model.kernel_.theta
			print(cost_cv[i])
		
		self.model.kernel_.theta = theta[np.argmin(cost_cv,axis=1)[0],:]
		"""

	def obj_func(self,hyperparams)->float:
		self.model.kernel_.theta = hyperparams
		prediction = self.model.predict(self.data_x)
		cost = cost_function(self.data_y,prediction)
		self.model.kernel_.theta = hyperparams

		return cost

	def optimizer(self):
		initial_theta = self.model.kernel_.theta
		optimalResult = scipy.optimize.minimize(self.obj_func, initial_theta, method='BFGS')
		theta_opt = optimalResult.x
		return theta_opt



def plot(x_train, y_train, x_test, predictions):
	# plotting data
	ax = plt.axes(projection='3d')
	ax.scatter3D(x_train[:,0], x_train[:,1], y_train, cmap='Greens');    
	ax.scatter3D(x_test[:,0], x_test[:,1], predictions, cmap='Reds')
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	plt.show()

def main():
#   HI there
	train_x_name = "train_x.csv"
	train_y_name = "train_y.csv"

	train_x = np.loadtxt(train_x_name, delimiter=',')
	train_y = np.loadtxt(train_y_name, delimiter=',')

	# load the test dateset
	test_x_name = "test_x.csv"
	test_x = np.loadtxt(test_x_name, delimiter=',')
	
	# choose smallest Euklidian distances between train_x and test_x
# 	distances = np.sqrt((train_x[:,0] - test_x[0,:])**2 + (train_x[:,1] - test_x[0,:])**2)	 
	

	M = Model()
	M.fit_model(train_x,train_y)
	prediction = M.predict(train_x)
	print(f"Cost on train_x data: {cost_function(train_y, prediction)}")

	prediction_test = M.predict(test_x)
	plot(M.data_x,M.data_y,test_x,prediction_test)


if __name__ == "__main__":
	main()
	print("Completed sucessfully")

