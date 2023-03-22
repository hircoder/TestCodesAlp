This code is a Python implementation of the K-means clustering algorithm for an interview assignment.
I used the following resources to learn about K-means clustering algorithm:
1. https://www.geeksforgeeks.org/k-means-clustering-introduction/
2. https://en.wikipedia.org/wiki/K-means_clustering
3. https://www.youtube.com/watch?v=4b5d3muPQmA

The objective of this code is to partition a set of data points into k clusters based on their similarity, where k is a given number.
This is done to ensure that data points within each cluster are as similar as possible and as dissimilar as possible from data points in other clusters.
To test the code, load datasets and perform K-means clustering on the data, specifying the number of clusters and maximum iterations.
The code assumes that the data is two-dimensional and the number of clusters is less than or equal to the number of data points.
The code visualizes the clusters and centroids in a scatter plot and saves the plot as an image file.
It also assumes that the data is stored in a CSV file with no header row and two columns: one for the X values and one for the Y values.
The data is loaded from the CSV file using Pandas and converted to a NumPy array.

The implementation assumes a simple K-means algorithm, meaning that initial centroids are randomly selected from the data points.
It also assumes the Euclidean distance is used to calculate the distance between data points and centroids.
As stated in the assignment, the expected outputs are the final set of clusters and their corresponding centroids.
Per the requirements, the K-means algorithm should have an interface similar to the one provided by scikit-learn.
Based on the scikit-learn documentation, the K-means algorithm has defined methods for fitting the model and predicting cluster labels.
Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
I have implemented the fit and predict methods, similar to the KMeans class in scikit-learn.
The fit method takes the data and the number of clusters as input and returns the model object.
The predict method takes the data as input and returns the predicted cluster labels for the data points.

To run this code:

1.  Open the terminal and navigate to the directory containing K-means2.py file.
2.  Run the following command: python K-means2.py <dataset_file> <n_clusters> <max_iterations>

How to try the algorithms on other datasets?
Give the path to .csv file of two dimensional points in the format mentioned above.

Deliverables:
1. K-means2.py
2. K-means2.ipynb
3. K-means2.html
