import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import minmax_scale
from scipy.special import softmax, expit
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

    def __init__(self, hexagonal_graph, learning_rate: float = 0.01):
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

    def triweight_function(self, t):
        return max(0, (1 - t**2)**3)
    
    def gaussian_rbf_kernel(self, x, y, sigma):
        return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))
    

    def gaussian_function(self, distance_to_bmu, neighborhood_radius):
        """
        Gaussian neighborhood function.
        """
        return np.exp(-(distance_to_bmu**2) / (2 * neighborhood_radius**2))
    
    def cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        similarity = dot_product / (norm_a * norm_b)
        return similarity

    def update_weights_with_gaussian_kernel(self, input_vector: np.ndarray, 
                                            bmu: int, epoch: int, max_epochs: int, 
                                            min_distance: float = 1.0, neighborhood_radius: float = 3, sigma: float = 2.0) -> None:
        """
        Update the weights of the SOM nodes based on the input vector and the best matching unit (BMU)
        using a Gaussian RBF kernel.

        Parameters:
        - input_vector (numpy.ndarray): Input vector used for weight updates.
        - bmu (int): Index of the best matching unit.
        - epoch (int): Current training epoch.
        - max_epochs (int): Maximum number of training epochs.
        - neighborhood_radius (float): Radius of the neighborhood for weight updates. Default is 1.4.
        - sigma (float): Width parameter of the Gaussian RBF kernel. Default is 2.0.
        """
        learning_rate = self.learning_rate * (1 - epoch / max_epochs)

        for node_id, weight in enumerate(self.weights):
            distance_to_bmu = np.linalg.norm(self.weights[bmu] - weight)  # Euclidean distance
            
            if distance_to_bmu >= min_distance:
            
                #normalized_distance = distance_to_bmu / neighborhood_radius
                # neighborhood_influence = self.triweight_function(normalized_distance)

                # Use the Gaussian as the neighborhood influence
                neighborhood_influence = self.gaussian_function(distance_to_bmu, neighborhood_radius)
            
                # Weight update equation using cosine similarity
                weight_update = learning_rate * neighborhood_influence * (input_vector - weight)
                self.weights[node_id] += weight_update
                
    def calculate_repulsive_force(self, node_id, other_node_id, repulsion_strength):
        weight = self.weights[node_id]
        other_weight = self.weights[other_node_id]
        distance_between_nodes = np.linalg.norm(weight - other_weight)
        repulsive_force = repulsion_strength / distance_between_nodes**2
        return repulsive_force * (weight - other_weight)
    
    def update_weights_with_repulsive_force(self, repulsion_strength):
        num_nodes = len(self.weights)

        for node_id in range(num_nodes):
         for other_node_id in range(num_nodes):
               if node_id != other_node_id:
                repulsive_update = self.calculate_repulsive_force(node_id, other_node_id, repulsion_strength)
                self.weights[node_id] += repulsive_update
                
    def quantization_error(self, input_data):
        errors = []
    
        for input_vector in input_data:
            bmu = self.find_best_matching_unit(input_vector)
            error = np.linalg.norm(input_vector - self.weights[bmu])  # Euclidean distance
            errors.append(error)
    
        return np.mean(errors)

    def fit(self, data: List[np.ndarray], epochs: int, min_distance: float, neighborhood_radius: float, repulsion_strength: float) -> None:
        """
        Train the SOM using the given data for a specified number of epochs.

        Parameters:
        - data (list): List of input vectors for training.
        - epochs (int): Number of training epochs.
        """
        min_distance = min_distance/epochs
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            min_distance = min_distance * (epoch + 1)
            random.shuffle(data.copy())
            for input_vector in data:
                bmu = self.find_best_matching_unit(input_vector)
                self.update_weights_with_gaussian_kernel(input_vector, bmu, epoch, epochs, min_distance=min_distance, neighborhood_radius=neighborhood_radius)
                
        error = self.quantization_error(data)
        print("Error = {}".format(error))
            
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


class RegSOM(SelfOrganizingMap):
    def __init__(self, hexagonal_graph, learning_rate: float = 0.01):
        super().__init__(hexagonal_graph, learning_rate)
        self.hexagonal_graph = hexagonal_graph
        
        # Initialize weights based on the shape of the target variable
        self.hexGraphNodes = np.array(hexagonal_graph.nodes())
        self.initialize_weights()
        
    def triweight_function(self, t):
        return np.maximum(0, (1 - t**2)**3)
    
    def initialize_weights(self):
        # Randomly initialize weights between 0 and 1 and scale them to the target shape
        self.weights = np.ones(self.hexGraphNodes.shape[0])
        
    def find_best_matching_unit(self, input_vector: np.ndarray) -> int:
        """
        Find the index of the node (unit) in the SOM whose node is closest to the given input vector.

        Parameters:
        - input_vector (numpy.ndarray): Input vector for which the best matching unit is to be found.

        Returns:
        - int: Index of the best matching unit.
        """
        distances = np.linalg.norm(self.hexGraphNodes - input_vector, axis=1)
        best_matching_unit_idx = np.argmin(distances)
        return best_matching_unit_idx
    
    def get_hex_neighborhood(self, idx: int):
        coordinates = tuple(self.hexGraphNodes[idx])
        neighbors = [np.array(x) for x in list(self.hexagonal_graph.neighbors(coordinates))]
        neighbors.append(np.array(coordinates))
        hex_neighborhood_indices = [np.where(np.all(self.hexGraphNodes == x, axis=1))[0][0] for x in neighbors]
        hex_neighborhood_values = np.take(self.weights, hex_neighborhood_indices)
        return hex_neighborhood_indices, hex_neighborhood_values, neighbors
    
    def update_weights_with_triWeight_function(self, input_vector, target_vector: np.ndarray, 
                                            bmu: int, epoch: int, max_epochs: int, 
                                            neighborhood_radius: float = 3) -> None:
        """
        Update the weights of the SOM nodes based on the input vector and the best matching unit (BMU)
        using a Gaussian RBF kernel.

        Parameters:
        - input_vector (numpy.ndarray): Input vector used for weight updates.
        - bmu (int): Index of the best matching unit.
        - epoch (int): Current training epoch.
        - max_epochs (int): Maximum number of training epochs.
        - neighborhood_radius (float): Radius of the neighborhood for weight updates. Default is 1.4.
        - sigma (float): Width parameter of the Gaussian RBF kernel. Default is 2.0.
        """
        learning_rate = self.learning_rate * (1 - epoch / max_epochs)
        
        # Get hexagonal neighborhood
        bmu_coordinates = self.hexGraphNodes[bmu]
        distances_to_bmu = np.linalg.norm(bmu_coordinates - self.hexGraphNodes, axis=1)
        distances_to_input = np.linalg.norm(input_vector - self.hexGraphNodes, axis=1)

        # Use the Gaussian RBF kernel as the neighborhood influence
        normalized_distance = distances_to_bmu / neighborhood_radius
        neighborhood_influence = self.triweight_function(normalized_distance)
        
        neighborhood_indices = np.where(neighborhood_influence > 0)[0]
        
        neighborhood_values = np.take(self.weights, neighborhood_indices)
        distances_to_input = np.take(distances_to_input, neighborhood_indices)**2
        
        distance_weights = neighborhood_values / distances_to_input
        
        # Apply IDW interpolation
        prediction = np.sum(distance_weights) / np.sum(1/distances_to_input)
        
        # Calculate the difference between the target value and the model's prediction
        prediction_error = target_vector - prediction
        
        for node_id, weight in zip(neighborhood_indices, distance_weights / (1 / distances_to_input)):
            # Weight update to minimize the prediction error
            weight_update = learning_rate * prediction_error * weight * target_vector
            
            self.weights[node_id] += weight_update
            
    def update_weights_w_neighborhood_influence_function(self, target_vector: np.ndarray, 
                                            bmu: int, epoch: int, max_epochs: int, 
                                            neighborhood_radius: float = 1.4) -> None:
        """
        Update the weights of the SOM nodes based on the input vector and the best matching unit (BMU)
        using a Gaussian RBF kernel.

        Parameters:
        - input_vector (numpy.ndarray): Input vector used for weight updates.
        - bmu (int): Index of the best matching unit.
        - epoch (int): Current training epoch.
        - max_epochs (int): Maximum number of training epochs.
        - neighborhood_radius (float): Radius of the neighborhood for weight updates. Default is 1.4.
        - sigma (float): Width parameter of the Gaussian RBF kernel. Default is 2.0.
        """
        learning_rate = self.learning_rate * (1 - epoch / max_epochs)
        
        # Get hexagonal neighborhood
        bmu_coordinates = self.hexGraphNodes[bmu]
        distances_to_bmu = np.linalg.norm(bmu_coordinates - self.hexGraphNodes, axis=1)

        # Use the Gaussian RBF kernel as the neighborhood influence
        normalized_distances = distances_to_bmu / neighborhood_radius
        neighborhood_influence = self.triweight_function(normalized_distances)
        
        for node_id, weight in enumerate(neighborhood_influence):
            # Weight update to minimize the prediction error
            weight_update = learning_rate * weight * target_vector
            
            self.weights[node_id] += weight_update
        
    def fine_tune_weights_w_bfs_IDW(self, input_vector: np.array, target_value: float,
                                      bmu: int, epoch: int, max_epochs: int) -> None:
        """
        Update the weights of the SOM nodes for regression using IDW interpolation.

        Parameters:
        - target_value (float): Target value to approximate.
        - bmu (int): Index of the best matching unit.
        - epoch (int): Current training epoch.
        - max_epochs (int): Maximum number of training epochs.
        """
        learning_rate = self.learning_rate * (1 - epoch / max_epochs)

        # Get BMU and it's immediate neighbors from the graph
        node_ids, node_vals, node_coordinates = self.get_hex_neighborhood(bmu)

        # Calculate distances between input vector and surrounding nodes
        distances = np.linalg.norm(node_coordinates - input_vector, axis=1) ** 2
        
        distances[distances == 0] = 1e-10
        distance_weights = node_vals/distances
        
        # Apply IDW interpolation
        prediction = np.sum(distance_weights) / np.sum(1/distances)
        
        # Calculate the difference between the target value and the model's prediction
        prediction_error = target_value - prediction
        
        distance_weights = distance_weights/(1/distances)
        #if np.abs(prediction_error) > 0.025:
        for node_id, weight in zip(node_ids, distance_weights):
            # Weight update to minimize the prediction error
            weight_update = learning_rate * weight * prediction_error * target_value

            self.weights[node_id] += weight_update
            
    def fit_sparse(self, data: List[np.ndarray], target: List[np.ndarray], epochs: int, r=3) -> None:
        """
        Train the SOM using the given data for a specified number of epochs.

        Parameters:
        - data (list): List of input vectors for training.
        - epochs (int): Number of training epochs.
        """
        
        # Round 1 of training, train using BFS + IDW
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            for (input_vector, target_vector) in zip(data, target):
                bmu = self.find_best_matching_unit(input_vector)
                self.fine_tune_weights_w_bfs_IDW(input_vector, target_vector, bmu, epoch, epochs)
        
        # Filter out nodes that were not updated using BFS
        self.weights[self.weights == 1.0] = np.nan

        # Round 2 of training, train using neighborhood influence function then fine tune using bfs + IDW
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            for (input_vector, target_vector) in zip(data, target):
                bmu = self.find_best_matching_unit(input_vector)
                self.update_weights_w_neighborhood_influence_function(target_vector, bmu, epoch, epochs, neighborhood_radius=r)
                self.fine_tune_weights_w_bfs_IDW(input_vector, target_vector, bmu, epoch, epochs)
                        
    def fit_dense(self, data: List[np.ndarray], target: List[np.ndarray], epochs: int, r=3, mask=0.0) -> None:
        """
        Train the SOM using the given data for a specified number of epochs.

        Parameters:
        - data (list): List of input vectors for training.
        - epochs (int): Number of training epochs.
        """
        
        # Round 1 of training, train using BFS + IDW
        #for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        #    for (input_vector, target_vector) in zip(data, target):
        #        bmu = self.find_best_matching_unit(input_vector)
        #        self.fine_tune_weights_w_bfs_IDW(input_vector, target_vector, bmu, epoch, epochs)
        
        # Filter out nodes that were not updated using BFS
        self.weights[self.weights == 1.0] = mask

        # Round 2 of training, train using neighborhood influence function then fine tune using bfs + IDW
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            for (input_vector, target_vector) in zip(data, target):
                bmu = self.find_best_matching_unit(input_vector)
                self.update_weights_w_neighborhood_influence_function(target_vector, bmu, epoch, epochs, neighborhood_radius=r)
                self.fine_tune_weights_w_bfs_IDW(input_vector, target_vector, bmu, epoch, epochs)
                
    def predict(self, data: List[np.ndarray], p: float = 1.0) -> List[float]:
        """
        Predict target values for the given input data.

        Parameters:
        - data (list): List of input vectors for prediction.

        Returns:
        - List[float]: Predicted target values.
        """
        predictions = []

        for input_vector in tqdm(data, desc="Predicting", unit="input"):
            # Find the best matching unit for the input vector
            bmu = self.find_best_matching_unit(input_vector)

            # Get BMU and its immediate neighbors from the graph
            node_ids, node_vals, node_coordinates = self.get_hex_neighborhood(bmu)

            # Calculate distances to BMU and its neighbors
            distances = np.linalg.norm(node_coordinates - input_vector, axis=1) ** p
            
            distances[distances == 0] = 1e-10            
            distance_weights = node_vals / distances

            # Apply IDW interpolation
            prediction = np.sum(distance_weights) / np.sum(1/distances)
            predictions.append(prediction)

        return np.array(predictions)
        

class SOM_Regression(SelfOrganizingMap):
    def __init__(self, input_size: int, hexagonal_graph, learning_rate: float = 0.01):
        super().__init__(input_size, hexagonal_graph, learning_rate)
        
    def update_weights_for_regression(self, input_vector: np.ndarray, target_value: float,
                                      bmu: int, epoch: int, max_epochs: int,
                                      neighborhood_radius: float = 1.5, sigma: float = 1.0) -> None:
        """
        Update the weights of the SOM nodes for regression.

        Parameters:
        - input_vector (numpy.ndarray): Input vector used for weight updates.
        - target_value (float): Target value to approximate.
        - bmu (int): Index of the best matching unit.
        - epoch (int): Current training epoch.
        - max_epochs (int): Maximum number of training epochs.
        - neighborhood_radius (float): Radius of the neighborhood for weight updates. Default is 1.4.
        - sigma (float): Width parameter of the Gaussian RBF kernel. Default is 1.0.
        """
        learning_rate = self.learning_rate * (1 - epoch / max_epochs)

        for node_id, weight in enumerate(self.weights):
            distance_to_bmu = np.linalg.norm(self.weights[bmu] - weight)  # Euclidean distance
            normalized_distance = distance_to_bmu / neighborhood_radius

            # Use the Gaussian RBF kernel as the neighborhood influence
            neighborhood_influence = self.gaussian_rbf_kernel(np.array([normalized_distance]), np.array([0]), sigma)

            # Calculate the difference between the target value and the model's prediction
            prediction = np.dot(weight, input_vector)  # Dot product as a simple linear regression model
            prediction_error = target_value - prediction

            # Weight update to minimize the prediction error
            weight_update = learning_rate * neighborhood_influence * prediction_error * input_vector

            self.weights[node_id] += weight_update
            
    def fit(self, data: List[np.ndarray], target: List[np.ndarray], epochs: int, min_distance: float, neighborhood_radius: float) -> None:
        """
        Train the SOM using the given data for a specified number of epochs.

        Parameters:
        - data (list): List of input vectors for training.
        - epochs (int): Number of training epochs.
        """
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            for (input_vector, target_vector) in zip(data, target):
                bmu = self.find_best_matching_unit(input_vector)
                self.update_weights_for_regression(input_vector, target_vector, bmu, epoch, epochs, min_distance=min_distance, neighborhood_radius=neighborhood_radius)
                
        error = self.quantization_error(data)
        print("Error = {}".format(error))
        
class StochasticSOM(RegSOM):
    def __init__(self, hexagonal_graph, learning_rate: float = 0.01):
        super().__init__(hexagonal_graph, learning_rate)
        self.variances = np.ones_like(self.weights)  # Initial variance for each node

    def compute_variances(self, data: List[np.ndarray]) -> np.ndarray:
        """
        Compute the empirical variance for each node based on the data points assigned to the BMU.
        For each data point, find the BMU and update the variance for the corresponding node.
        """
        squared_differences = np.zeros_like(self.weights)  # Sum of squared differences
        node_counts = np.zeros_like(self.weights)  # Count of data points assigned to each node
        
        # Iterate over each data point in the training set
        for input_vector in data:
            # Find the Best Matching Unit (BMU) for each data point
            bmu_idx = self.find_best_matching_unit(input_vector)
            
            # Get the weight vector of the BMU node
            bmu_weight = self.weights[bmu_idx]
            
            # Compute the squared distance between the data point and the BMU weight vector
            squared_distance = np.sum((input_vector - bmu_weight) ** 2)
            
            # Accumulate the squared distance and count the data points assigned to the BMU
            squared_differences[bmu_idx] += squared_distance
            node_counts[bmu_idx] += 1
        
        # Compute the variance for each node (BMU) based on the accumulated squared differences
        for i in range(self.weights.shape[0]):
            if node_counts[i] > 0:
                # Variance is the average of squared distances for each node
                self.variances[i] = squared_differences[i] / node_counts[i]
            else:
                # If no data points are assigned to the node, set variance to 0
                self.variances[i] = 0
                
        
    def fit_dense(self, data: List[np.ndarray], target: List[np.ndarray], epochs: int, r=3, mask=0.0) -> None:
        """
        Train the SOM using the given data for a specified number of epochs.

        Parameters:
        - data (list): List of input vectors for training.
        - epochs (int): Number of training epochs.
        """
        
        # Filter out nodes that were not updated using BFS
        self.weights[self.weights == 1.0] = mask

        # Round 2 of training, train using neighborhood influence function then fine tune using bfs + IDW
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            for (input_vector, target_vector) in zip(data, target):
                bmu = self.find_best_matching_unit(input_vector)
                self.update_weights_w_neighborhood_influence_function(target_vector, bmu, epoch, epochs, neighborhood_radius=r)
                self.fine_tune_weights_w_bfs_IDW(input_vector, target_vector, bmu, epoch, epochs)
                
        # After training, compute the variances based on BMU assignments
        self.compute_variances(data)

