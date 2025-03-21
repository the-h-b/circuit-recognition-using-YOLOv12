from sklearn.cluster import KMeans
import numpy as np

def cluster_components(detections, intersections):
    """Clusters components and nodes using K-means."""
    points = []
    component_labels = {}  # Map components to cluster labels

    # Add bounding box centers to points
    for index, row in detections.iterrows():
        x_center = int((row['xmin'] + row['xmax']) / 2)
        y_center = int((row['ymin'] + row['ymax']) / 2)
        points.append([x_center, y_center])
        component_labels[index] = None #Initialize component labels

    # Add intersection points to points
    for x, y in intersections:
        points.append([x, y])

    if not points:
        return {}, {}

    kmeans = KMeans(n_clusters=len(detections) + 1, random_state=0) # Add 1 to accomodate extra node clusters.
    labels = kmeans.fit_predict(points)

    # Assign cluster labels to component indexes
    component_index = 0
    for index in component_labels:
        component_labels[index] = labels[component_index]
        component_index += 1

    return labels, component_labels