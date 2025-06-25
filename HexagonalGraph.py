import numpy as np
import networkx as nx
from itertools import combinations
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

class HexagonalGrid:
    """
    Represents a hexagonal grid in 2D space.

    Parameters:
    - x_range (tuple): Range of x-coordinates.
    - y_range (tuple): Range of y-coordinates.
    - map_size (tuple): Size of the grid in terms of the number of hexagons in the x and y directions.
    - ratio (float): Ratio of the hexagon's width to its height.
    """

    def __init__(self, x_range, y_range, map_size=(10, 10), ratio=np.sqrt(3)/2):
        self._ratio = ratio
        self.x_range = x_range
        self.y_range = y_range
        self._x = map_size[0]
        self._y = map_size[1]

    def generate_2D_grid(self):
        """
        Generate a 2D hexagonal grid.

        Returns:
        - xv (numpy.ndarray): X-coordinates of the grid.
        - yv (numpy.ndarray): Y-coordinates of the grid.
        """
        x_range = np.linspace(self.x_range[0], self.x_range[1], self._x)
        y_range = np.linspace(self.y_range[0], self.y_range[1], self._y)
        xv, yv = np.meshgrid(x_range, y_range, sparse=False, indexing='xy')
        xv = xv * self._ratio
        xv[::2, :] += self._ratio/2
        return yv, xv

class HexagonalGraph(HexagonalGrid):
    """
    Represents a hexagonal graph based on a hexagonal grid.

    Parameters:
    - ratio (float): Ratio of the hexagon's width to its height.
    - N_X (int): Number of hexagons in the x direction.
    - N_Y (int): Number of hexagons in the y direction.
    - x_range (tuple): Range of x-coordinates.
    - y_range (tuple): Range of y-coordinates.
    - distance_threshold (float): Maximum distance for considering edges between nodes.
    """

    def __init__(self, ratio=np.sqrt(3)/2, N_X=12, N_Y=11, x_range=(0, 1), y_range=(0, 1), distance_threshold=1.4):
        super().__init__(map_size=(N_X, N_Y), ratio=ratio, x_range=x_range, y_range=y_range)
        self.N_X = N_X
        self.N_Y = N_Y
        self.distance_threshold = distance_threshold
        self._x, self._y = self.generate_2D_grid()

    def create_graph(self):
        """
        Create a hexagonal graph based on the generated grid.

        Returns:
        - G (networkx.Graph): Hexagonal graph.
        """
        G = nx.Graph()
        for point in zip(self._x.flatten(), self._y.flatten()):
            G.add_node(point)

        for pair in combinations(G.nodes(), 2):
            node1, node2 = pair
            distance = euclidean(node1, node2)
            if distance <= self.distance_threshold:
                G.add_edge(node1, node2)

        return G

def visualize_graph(G):
    """
    Visualize a hexagonal graph.

    Parameters:
    - G (networkx.Graph): Hexagonal graph to be visualized.
    """
    pos = {node: node for node in G.nodes()}
    nx.draw(G, pos, with_labels=False, font_size=8, font_color="black", node_size=30, node_color="skyblue", font_weight="bold", edge_color="gray", linewidths=0.5)
    plt.show()

