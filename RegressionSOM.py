import numpy as np
from tqdm import tqdm
import random
from SelfOrganizingMap import SelfOrganizingMap

class RegressionSOM(SelfOrganizingMap):
    """
    Implementation of a Self-Organizing Map (SOM) for regression tasks.

    Parameters:
    - input_size (int): Size of the input vectors.
    - hexagonal_graph (HexagonalGraph): Hexagonal graph representing the SOM topology.
    - learning_rate (float): Learning rate for weight updates during training. Default is 0.01.
    """

    def __init__(self, input_size, hexagonal_graph, learning_rate=0.01):
        super().__init__(input_size, hexagonal_graph, learning_rate)

    def fit(self, data, targets, epochs):
        """
        Train the RegressionSOM using the given data and targets for a specified number of epochs.

        Parameters:
        - data (list): List of input vectors for training.
        - targets (list): List of target values corresponding to the input vectors.
        - epochs (int): Number of training epochs.
        """
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            shuffled_data = list(zip(data, targets))
            random.shuffle(shuffled_data)
            for input_vector, target in shuffled_data:
                bmu = self.find_best_matching_unit(input_vector)
                self.update_weights(input_vector, bmu, epoch, epochs, target)  # Pass target to update_weights

    def update_weights(self, input_vector, bmu, epoch, max_epochs, neighborhood_radius=1.4, target=None):
        """
        Update the weights of the SOM nodes based on the input vector, the best matching unit (BMU), and target values.

        Parameters:
        - input_vector (numpy.ndarray): Input vector used for weight updates.
        - bmu (int): Index of the best matching unit.
        - epoch (int): Current training epoch.
        - max_epochs (int): Maximum number of training epochs.
        - neighborhood_radius (float): Radius of the neighborhood for weight updates. Default is 1.4.
        - target (float or None): Target value for regression. If None, the input_vector is used for updates.
        """
        learning_rate = self.learning_rate * (1 - epoch / max_epochs)

        for node_id, weight in enumerate(self.weights):
            distance = np.linalg.norm(self.weights[bmu] - weight)  # Euclidean distance
            neighborhood_influence = np.exp(-(distance**2) / (2 * neighborhood_radius**2))
            
            if target is not None:
                weight_update = learning_rate * neighborhood_influence * (target - weight)
            else:
                weight_update = learning_rate * neighborhood_influence * (input_vector - weight)
                
            self.weights[node_id] += weight_update
            
    def predict(self, input_vector):
        """
        Predict the regression output for a given input vector.

        Parameters:
        - input_vector (numpy.ndarray): Input vector for regression prediction.

        Returns:
        - numpy.ndarray: Predicted regression output.
        """
        bmu = self.find_best_matching_unit(input_vector)
        return self.weights[bmu]

