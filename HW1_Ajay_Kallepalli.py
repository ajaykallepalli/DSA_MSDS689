import numpy as np
class KMeans:
    def __init__(self, n_clusters=3, max_iterations=100, random_state=None):
        # Add input validation
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, data):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        m, n = data.shape
        
        # Initialize centroids more strategically
        # Choose random points from the data
        random_indices = np.random.choice(m, self.n_clusters, replace=False)
        self.centroids = data[random_indices].astype(np.float64)
        
        for _ in range(self.max_iterations):
            old_centroids = self.centroids.copy()
            
            # Calculate distances using broadcasting
            distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            for k in range(self.n_clusters):
                cluster_points = data[self.labels == k]
                if len(cluster_points) > 0:
                    self.centroids[k] = np.mean(cluster_points, axis=0)
            
            # Check for convergence
            if np.allclose(old_centroids, self.centroids):
                break
                
        return self

    def predict(self, data):
        if self.centroids is None:
            raise ValueError("You must call fit before predicting!")
        
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def get_centroids(self):
        """
        Get the current centroids after fitting the algorithm.

        Returns:
        - centroids: Numpy array representing the centroids of clusters.
        """
        if self.centroids is None:
            raise ValueError("You must call fit before getting centroids!")
        return self.centroids
