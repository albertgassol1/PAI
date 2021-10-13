import os
import typing

from sklearn.gaussian_process.kernels import *
from sklearn.kernel_approximation import Nystroem
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from scipy.optimize import minimize, dual_annealing
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.cluster.vq import vq


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0

k1 = Real(low=-100.0, high=100.0, prior='uniform', name='k1')
k2 = Real(low=-100.0, high=100.0, prior='uniform', name='k2')
k3 = Real(low=-100.0, high=100.0, prior='uniform', name='k3')
dimensions = [k1, k2, k3]

# Mattern 2.5

class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        # self.rng = np.random.default_rng(seed=0)

        # TODO: Add custom initialization for your model here if necessary

        # Kernels
        # self.kernel = ConstantKernel() + RBF() + WhiteKernel()
        # self.kernel = RBF(10, (1e-2, 1e2))
        self.kernel = Matern(nu=1.5)
        # We use kmeans to reduce number of points, this variable determines the number of n_clusters
        self.n_clusters = 250
        self.neighbors = 25

        # Initialize Nystrom transform
        self.nystrom_map = Nystroem(
            random_state=1,
            n_components=1
        )

        # GPR object
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9, normalize_y=True, random_state=0)


    def preprocess_data(self, train_x: np.ndarray, train_y: np.ndarray):
        data = np.column_stack((train_x, train_y))
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        kmeans.fit(data)
        k_pred = kmeans.predict(data) 
        dist = kmeans.transform(data)
        result = np.zeros([self.n_clusters*self.neighbors, 3])

        indices = []

        for n in range(self.n_clusters):
            indices_cluster = [i for i, x in enumerate(k_pred==n) if x]
            N = min([len(indices_cluster), self.neighbors])
            for j in range(N):   
                indices.append(indices_cluster[int(np.argmin(dist[indices_cluster, n]))])
                indices_cluster = np.delete(indices_cluster, np.argmin(dist[indices_cluster, n]))


        

        # return kmeans.cluster_centers_
        return indices

    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        gp_mean = np.zeros(x.shape[0], dtype=float)
        gp_std = np.zeros(x.shape[0], dtype=float)

        gp_mean, gp_std = self.gp.predict(x, return_std=True)

        # TODO: Use the GP posterior to form your predictions here
        # predictions = gp_mean
        predictions = np.where((gp_mean <= THRESHOLD) & (gp_mean + gp_std/2 > THRESHOLD), THRESHOLD, gp_mean)

        return predictions, gp_mean, gp_std
        

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        # Save data
        self.train_x = train_x
        self.train_y = train_y

        # TODO: Fit your model here
        # training_data = self.preprocess_data(train_x, train_y)
        # self.gp.fit(training_data[:, 0:2], training_data[:, -1])
        training_data_indices = self.preprocess_data(train_x, train_y)
        self.gp.fit(train_x[training_data_indices], train_y[training_data_indices])
        # self.gp.fit(training_data[:, 0:2], training_data[:, -1])

        # Minimize cost function to obtein hyperparameters
        # theta = minimize(self.optimization_function, self.gp.kernel_.theta, args=([train_x, train_y]), method='BFGS')
        
        # default_parameters = [self.gp.kernel_.theta[0], self.gp.kernel_.theta[1], self.gp.kernel_.theta[2]]
        # theta = gp_minimize(func=self.optimization_function,
        #                         dimensions=dimensions,
        #                         x0=default_parameters)
        # lb = [-100] * len(hyperparameters)
        # ub = [100] * len(hyperparameters)
        # theta = dual_annealing(self.optimization_function_annealing, bounds=list(zip(lb, ub)), x0=hyperparameters)
        # self.gp.kernel_.theta = theta.x

    # @use_named_args(dimensions=dimensions)
    def optimization_function(self, k):
       
        train_x =self.train_x
        train_y =self.train_y
        # self.gp.fit(self.train_x, self.train_y)
        self.gp.kernel_.theta = [k[0], k[1], k[2]]
        # self.gp.fit(self.train_x, self.train_y)
        prediction = self.gp.predict(train_x)

        assert train_y.ndim == 1 and prediction.ndim == 1 and train_y.shape == prediction.shape

        # Unweighted cost
        cost = (train_y - prediction) ** 2
        weights = np.zeros_like(cost)

        # Case i): overprediction
        mask_1 = prediction > train_y
        weights[mask_1] = COST_W_OVERPREDICT

        # Case ii): true is above threshold, prediction below
        mask_2 = (train_y >= THRESHOLD) & (prediction < THRESHOLD)
        weights[mask_2] = COST_W_THRESHOLD

        # Case iii): everything else
        mask_3 = ~(mask_1 | mask_2)
        weights[mask_3] = COST_W_NORMAL

        # Weigh the cost and return the average
        return np.mean(cost * weights)

    def optimization_function_annealing(self, hyperparameters):
        
        self.gp.kernel_.theta = hyperparameters
        prediction = self.gp.predict(self.train_x)
        cost = cost_function(self.train_y, prediction)
        self.gp.kernel_.theta = hyperparameters

        return cost
    

def cost_function(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case i): overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case ii): true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case iii): everything else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fit_model(train_x, train_y)

    # Predict on the test features
    print('Predicting on test features')
    predicted_y = model.predict(test_x)
    print(predicted_y)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
