import pytest
import numpy as np
from HW1_Ajay_Kallepalli import KMeans

@pytest.fixture
def simple_dataset():
    """Create a simple dataset with clear clusters for testing"""
    np.random.seed(42)
    cluster1 = np.random.normal(0, 0.5, (100, 2))
    cluster2 = np.random.normal(4, 0.5, (100, 2))
    cluster3 = np.random.normal([2, 4], 0.5, (100, 2))
    return np.vstack([cluster1, cluster2, cluster3])

def test_kmeans_initialization():
    """Test KMeans initialization with different parameters"""
    kmeans = KMeans(n_clusters=3, max_iterations=100, random_state=42)
    assert kmeans.n_clusters == 3
    assert kmeans.max_iterations == 100
    assert kmeans.random_state == 42
    assert kmeans.centroids is None
    assert kmeans.labels is None

def test_kmeans_fit_predict(simple_dataset):
    """Test basic fitting and prediction functionality"""
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(simple_dataset)
    
    # Check if centroids were computed
    assert kmeans.centroids is not None
    assert kmeans.centroids.shape == (3, 2)
    
    # Check if labels were assigned
    labels = kmeans.predict(simple_dataset)
    assert len(labels) == len(simple_dataset)
    assert len(np.unique(labels)) <= 3

def test_kmeans_convergence():
    """Test if KMeans converges with simple, well-separated data"""
    # Create three distinct points
    data = np.array([[0, 0], [10, 10], [20, 20]])
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data)
    
    # With distinct points, centroids should exactly match input points
    assert len(kmeans.centroids) == 3
    assert np.allclose(np.sort(kmeans.centroids, axis=0), np.sort(data, axis=0))

def test_kmeans_empty_clusters():
    """Test handling of potential empty clusters"""
    # Create data with only two distinct points but ask for three clusters
    data = np.array([[0, 0], [0, 0], [10, 10], [10, 10]])
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data)
    
    # Should still complete without errors
    assert kmeans.centroids is not None
    labels = kmeans.predict(data)
    assert len(np.unique(labels)) <= 3

def test_kmeans_random_state():
    """Test reproducibility with random_state"""
    data = np.random.rand(100, 2)
    
    kmeans1 = KMeans(n_clusters=3, random_state=42)
    kmeans2 = KMeans(n_clusters=3, random_state=42)
    
    centroids1 = kmeans1.fit(data).centroids
    centroids2 = kmeans2.fit(data).centroids
    
    assert np.allclose(centroids1, centroids2)

def test_input_validation():
    """Test error handling for invalid inputs"""
    kmeans = KMeans(n_clusters=3)
    
    # Test invalid number of clusters
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)
    
    with pytest.raises(ValueError):
        KMeans(n_clusters=-1)
    
    # Test invalid input data
    with pytest.raises(ValueError):
        kmeans.fit(np.array([]))  # Empty array
    
    with pytest.raises(ValueError):
        kmeans.fit(np.array([[1]]))  # Single sample

def test_predict_without_fit():
    """Test error handling when predict is called before fit"""
    kmeans = KMeans(n_clusters=3)
    data = np.random.rand(10, 2)
    
    with pytest.raises(ValueError):
        kmeans.predict(data)

def test_different_dimensions():
    """Test KMeans with different dimensional data"""
    # Test with 1D data
    data_1d = np.random.rand(100, 1)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(data_1d)
    assert kmeans.centroids.shape == (2, 1)
    
    # Test with 3D data
    data_3d = np.random.rand(100, 3)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(data_3d)
    assert kmeans.centroids.shape == (2, 3)

def test_prediction_shape_mismatch():
    """Test error handling for prediction with wrong dimensions"""
    kmeans = KMeans(n_clusters=3, random_state=42)
    train_data = np.random.rand(100, 2)
    kmeans.fit(train_data)
    
    # Try to predict with different dimensions
    test_data = np.random.rand(10, 3)
    with pytest.raises(ValueError):
        kmeans.predict(test_data)