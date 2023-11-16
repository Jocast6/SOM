import numpy as np
import random
from tqdm import tqdm
from typing import List

class SelfOrganizingMap:
    """
    Implementation of a Self-Organizing Map (SOM).

    Parameters:
    - input_size (int): Size of the input vectors.
    - hexagonal_graph (HexagonalGraph): Hexagonal graph representing the SOM topology.
    - learning_rate (float): Learning rate for weight updates during training. Default is 0.01.
    """

    def __init__(self, input_size: int, hexagonal_graph, learning_rate: float = 0.01):
        self.input_size = input_size
        self.learning_rate = learning_rate

        # Use the HexagonalGraph instead of hexagonal_lattice_graph
        self.som_graph = hexagonal_graph

        # Initialize weights for each node
        self.weights = np.array(hexagonal_graph.nodes)

    def find_best_matching_unit(self, input_vector: np.ndarray) -> int:
        """
        Find the index of the node (unit) in the SOM whose weight is closest to the given input vector.

        Parameters:
        - input_vector (numpy.ndarray): Input vector for which the best matching unit is to be found.

        Returns:
        - int: Index of the best matching unit.
        """
        distances = np.linalg.norm(self.weights - input_vector, axis=1)
        best_matching_unit = np.argmin(distances)
        return best_matching_unit

    def update_weights(self, input_vector: np.ndarray, 
                       bmu: int, epoch: int, max_epochs: int, 
                       neighborhood_radius: float = 1.4, min_distance: float = 1) -> None:
        """
        Update the weights of the SOM nodes based on the input vector and the best matching unit (BMU).

        Parameters:
        - input_vector (numpy.ndarray): Input vector used for weight updates.
        - bmu (int): Index of the best matching unit.
        - epoch (int): Current training epoch.
        - max_epochs (int): Maximum number of training epochs.
        - neighborhood_radius (float): Radius of the neighborhood for weight updates. Default is 1.4.
        - min_distance (float): Minimum allowed distance for weight updates. Default is 0.5.
        """
        learning_rate = self.learning_rate * (1 - epoch / max_epochs)

        for node_id, weight in enumerate(self.weights):
            distance_to_bmu = np.linalg.norm(self.weights[bmu] - weight)  # Euclidean distance
            
            # Check if the distance is greater than the minimum allowed distance
            if distance_to_bmu >= min_distance:
                neighborhood_influence = np.exp(-(distance_to_bmu**2) / (2 * neighborhood_radius**2))
                weight_update = learning_rate * neighborhood_influence * (input_vector - weight)
                self.weights[node_id] += weight_update

    def fit(self, data: List[np.ndarray], epochs: int) -> None:
        """
        Train the SOM using the given data for a specified number of epochs.

        Parameters:
        - data (list): List of input vectors for training.
        - epochs (int): Number of training epochs.
        """
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            random.shuffle(data.copy())
            for input_vector in data:
                bmu = self.find_best_matching_unit(input_vector)
                self.update_weights(input_vector, bmu, epoch, epochs)

    def assign_labels(self, data: List[np.ndarray]) -> List[int]:
        """
        Assign labels to input data based on the best matching units.

        Parameters:
        - data (list): List of input vectors.

        Returns:
        - list: List of labels corresponding to the best matching units for each input vector.
        """
        labels = [self.find_best_matching_unit(input_vector) for input_vector in data]
        return labels

    def predict(self, input_vector: np.ndarray, num_clusters: int) -> np.ndarray:
        """
        Predict cluster assignments for a given input vector.

        Parameters:
        - input_vector (numpy.ndarray): Input vector for prediction.
        - num_clusters (int): Number of clusters to assign.

        Returns:
        - numpy.ndarray: Array of indices representing the predicted cluster assignments.
        """
        bmu = self.find_best_matching_unit(input_vector)

        # Assign each node to the closest cluster based on Euclidean distance
        distances_to_bmu = np.linalg.norm(self.weights - self.weights[bmu], axis=1)
        cluster_assignments = np.argsort(distances_to_bmu)[:num_clusters]

        return cluster_assignments


