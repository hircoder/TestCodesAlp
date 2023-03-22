# This code is the python implementation of K-means clustering algorithm for interview assignment.
# I used the following resources to learn about K-means clustering algorithm:
# https://www.geeksforgeeks.org/k-means-clustering-introduction/
# https://en.wikipedia.org/wiki/K-means_clustering
# https://www.youtube.com/watch?v=4b5d3muPQmA

# In this code we aim to partition a set of data points into k clusters based on the their similarity, where k is a given number.
# We do so to keep data points for each cluster as similar as possible and as different as possible from data points in other clusters.
# You can test the code by loading datasets and perform K-means clustering on the data, specifying the number of clusters and maximum iterations.
# How to run this code:
# 1. Open the terminal and navigate to the directory containing this file.
# 2. Run the following command: python kmeans.py <dataset_file> <n_clusters> <max_iterations>

# Used for randomly selecting initial centroids.
import random

# Used for handling arrays and calculating distances.
import numpy as np

# Used for loading the data from the CSV file.
import pandas as pd

# Used for visualizing the clusters and centroids.
import matplotlib.pyplot as plt

# Used for parsing command-line arguments.
import argparse

# Used for handling file path validation and exiting the script when needed.
import os.path
import sys


def generate_colors(num_colors):
    """
    Generate a list of RGB colors with a specified number of colors.

    Parameters:
    num_colors (int): The number of colors to generate.

    Returns:
    list: The list of generated RGB colors.
    """
    random_colors = set()

    while len(random_colors) < num_colors:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        random_colors.add(color)

    return list(random_colors)


def visualize_clusters(data, cluster_labels, centroids, num_clusters):
    """
    Visualize the clusters and centroids in a scatter plot.

    Parameters:
    data (numpy.ndarray): The dataset used for clustering.
    cluster_labels (numpy.ndarray): The predicted cluster labels for the data points.
    centroids (numpy.ndarray): The centroids of the clusters.
    num_clusters (int): The number of clusters.
    """

    # Generate a list of colors for each cluster
    colors = generate_colors(num_clusters)

    # Iterate over each cluster and plot its data points with the corresponding color
    for i in range(num_clusters):
        plt.scatter(
            data[
                cluster_labels == i, 0
            ],  # X values of the data points in the current cluster
            data[
                cluster_labels == i, 1
            ],  # Y values of the data points in the current cluster
            s=50,  # size of the data points
            c=colors[i],  # color of the data points in the current cluster
            label=f"Cluster {i+1}",  # label for the current cluster
        )

    # Plot the centroids as stars
    plt.scatter(
        centroids[:, 0],  # X values of the centroids
        centroids[:, 1],  # Y values of the centroids
        marker="+",  # marker style for the centroids
        s=100,  # size of the centroids
        c="#000000",  # color of the centroids
        label="Centroids",  # label for the centroids
    )

    # Add axis labels and a legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # Show the plot
    plt.show()

    # Save the plot as an image file
    plt.savefig(f"Clustered_S{len(data)}_C{num_clusters}.png")


def load_data(file_name):
    """
    Load the data from the CSV file using Pandas.

    Parameters:
    file_name (str): The name of the CSV file containing the data.

    Returns:
    numpy.ndarray: The loaded data as a NumPy array.
    """
    data = pd.read_csv(file_name, header=None)

    return data.values


# Implementation of K-means clustring algorithm
# K-means algorithm is an unsupervised learning technique used for clustering data.
# The process of k-means clustering involves the following steps:
# I. Initialization: Choose the number of clusters, K, and randomly initialize K centroids.
# II. Assignment: Assign each data point to the nearest centroid based on the Euclidean distance between the data point and the centroid.
# III. Update: Recalculate the centroids of each cluster based on the mean of the data points assigned to it.
# IV. Repeat: Repeat steps 2 and 3 until the centroids no longer move significantly or a maximum number of iterations is reached.
# V. Output: The output of the algorithm is the final set of K clusters and their corresponding centroids.
class Kmeans:
    def __init__(self, n_clusters=5, max_iterations=300):
        """Initialize the number of clusters and maximum iterations."""
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def fit(self, X):
        """Fit the model to the given data."""
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iterations):
            clusters = self._create_clusters(X, self.centroids)
            prev_centroids = self.centroids
            self.centroids = self._calculate_centroids(clusters, X)

            if self._is_converged(prev_centroids, self.centroids):
                break

    def predict(self, X):
        """Predict the cluster assignments for the given data."""
        clusters = self._create_clusters(X, self.centroids)
        y_pred = np.zeros(X.shape[0], dtype=int)

        for i, cluster in enumerate(clusters):
            y_pred[cluster] = i

        return y_pred, self.centroids

    def _initialize_centroids(self, X):
        """Randomly select n_clusters data points from the dataset as initial centroids."""
        random_indices = random.sample(range(X.shape[0]), self.n_clusters)
        return X[random_indices]

    def _create_clusters(self, X, centroids):
        """Assign each data point to the closest centroid."""
        clusters = [[] for _ in range(self.n_clusters)]

        for i, x in enumerate(X):
            distances = [np.linalg.norm(x - c) for c in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(i)

        return clusters

    def _calculate_centroids(self, clusters, X):
        """Calculate the new centroids as the mean of the data points in each cluster."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid

        return centroids

    def _is_converged(self, prev_centroids, centroids):
        """Check if the previous centroids are equal to the current centroids."""
        return np.array_equal(prev_centroids, centroids)


# Test
# Function to load and process the selected dataset with the K-means algorithm
def load_and_process_kmeans(data_file, n_clusters, max_iterations):
    """
    Loads the selected dataset and processes it with the K-means algorithm.
    Initializes the K-means object, fits the model, predicts cluster assignments and centroids, and visualizes the clusters and centroids.

    Parameters:
    data_file (str): The name of the CSV file containing the data.
    n_clusters (int): The number of clusters to be created.
    max_iterations (int): The maximum number of iterations to be performed.

    Returns:
    None
    """
    try:
        # Load the selected dataset
        data = load_data(data_file)
    except FileNotFoundError:
        # Handle the case when the specified dataset file is not found
        print(f"Error: File '{data_file}' not found.")
        sys.exit(1)

    # Process the selected dataset with the K-means algorithm
    # Initialize the K-means object with the number of clusters and maximum iterations
    kmeans = Kmeans(n_clusters, max_iterations)

    # Fit the Kmeans model to the data
    kmeans.fit(data)

    # Predict cluster assignments and centroids
    cluster_pred, centroids = kmeans.predict(data)

    # Visualize the clusters and centroids
    visualize_clusters(data, cluster_pred, kmeans.centroids, n_clusters)


# Main function to parse command-line arguments and run the K-means clustering
# This functions relies on the following external libraries:
# argparse for parsing command-line arguments.
# os and sys for handling file path validation and exiting the script when needed.
def main():
    """
    Parses command-line arguments and runs the K-means clustering.

    Parameters:
    None

    Returns:
    None
    """
    # Set up command-line argument parsing to accept user input for the dataset, number of clusters, and maximum iterations.

    parser = argparse.ArgumentParser(description="K-means Clustering Example")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data_C17N_5000.csv",
        help="The name of the dataset file to be used (default: data_C17N_5000.csv).",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=17,
        help="The number of clusters to be generated (default: 17).",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=300,
        help="The maximum number of iterations for the K-means algorithm (default: 300).",
    )

    args = parser.parse_args()

    # Check if the provided dataset is valid
    if not os.path.exists(args.dataset):
        print(f"Error: Invalid dataset name. Please write the path to a valid dataset.")
        sys.exit(1)

    n_clusters = args.num_clusters
    data_file = args.dataset

    # Inform the user about the selected dataset and settings
    print(
        f"Processing dataset: {data_file} with {n_clusters} clusters and {args.max_iterations} max iterations"
    )

    # Load and process the selected dataset with the K-means algorithm
    load_and_process_kmeans(data_file, n_clusters, args.max_iterations)


# Check if the script is being run as the main module and call the main function
if __name__ == "__main__":
    main()
