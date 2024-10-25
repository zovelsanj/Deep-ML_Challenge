"""PROBLEM: K-Means Clustering (medium)
Your task is to write a Python function that implements the k-Means clustering algorithm. 
This function should take specific inputs and produce a list of final centroids. 
k-Means clustering is a method used to partition n points into k clusters. 
The goal is to group similar points together and represent each group by its center (called the centroid).

Function Inputs:
points: A list of points, where each point is a tuple of coordinates (e.g., (x, y) for 2D points)
k: An integer representing the number of clusters to form
initial_centroids: A list of initial centroid points, each a tuple of coordinates
max_iterations: An integer representing the maximum number of iterations to perform
Function Output:
A list of the final centroids of the clusters, where Each centroid is rounded to the nearest fourth decimal.

Example:
    input: points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], k = 2, initial_centroids = [(1, 1), (10, 1)], max_iterations = 10
    output: [(1, 2), (10, 2)]
    reasoning: Given the initial centroids and a maximum of 10 iterations, the points are clustered around these points, and the centroids are updated to the mean of the assigned points, 
    resulting in the final centroids which approximate the means of the two clusters.
    The exact number of iterations needed may vary, but the process will stop after 10 iterations at most.
"""

import numpy as np

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    def euclidean_distance(p1, p2):
        return np.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

    def calculate_mean(cluster):
        return tuple(round(sum(coord) / len(coord), 4) for coord in zip(*cluster))

    final_centroids = initial_centroids
    
    for iteration in range(max_iterations):
        clusters = {i: [] for i in range(k)}
        for point in points:
            closest_centroid = min(range(k), key=lambda c: euclidean_distance(point, final_centroids[c]))
            clusters[closest_centroid].append(point)
        
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                new_centroid = calculate_mean(clusters[i])
            else:
                new_centroid = final_centroids[i]
            new_centroids.append(new_centroid)

        if new_centroids == final_centroids:
            break

        final_centroids = new_centroids
    
    return final_centroids

if __name__ == "__main__":
    print(k_means_clustering([(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], k=2, initial_centroids=[(1, 1), (10, 1)], max_iterations=10))
    print(k_means_clustering([(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)], 2, [(1, 1, 1), (10, 10, 10)], 10))
    print(k_means_clustering([(1, 1), (2, 2), (3, 3), (4, 4)], 1, [(0,0)], 10))
