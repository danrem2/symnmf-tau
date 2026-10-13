import math

EPSILON = 1e-4
MAX_ITER = 300


def get_distance(p, q):
    """Euclidean distance between two points."""
    distance = 0.0
    for i in range(len(p)):
        distance += (p[i] - q[i]) ** 2
    return math.sqrt(distance)


def assign_clusters(points, centroids):
    """Assign every point to its closest centroid."""
    labels = []
    for point in points:
        closest, smallest = 0, float("inf")
        for index, centroid in enumerate(centroids):
            distance = get_distance(point, centroid)
            if distance < smallest:
                closest, smallest = index, distance
        labels.append(closest)
    return labels


def kmeans(points, k):
    """Cluster points with K-means and return the resulting labels."""
    centroids = [points[i][:] for i in range(k)]
    for _ in range(MAX_ITER):
        labels = assign_clusters(points, centroids)
        movement = 0.0
        for index in range(k):
            members = [p for p, label in zip(points, labels) if label == index]
            if not members:
                continue
            updated = [sum(values) / len(members) for values in zip(*members)]
            movement = max(movement, get_distance(updated, centroids[index]))
            centroids[index] = updated
        if movement < EPSILON:
            break
    return assign_clusters(points, centroids)
