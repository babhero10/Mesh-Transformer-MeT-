import numpy as np
from scipy.cluster.hierarchy import ward, fcluster
from utils import load_mesh, get_mesh_adjacency_and_degree_matrices, get_eigenvectors, compute_triangle_normals
from visualization import visualize_clustering_feature, visualize_positional_encoding

def get_clustering_features(mesh, lambda_val=8):
    # Convert Open3D mesh to NumPy arrays
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Compute features for each triangle (centroid in this case)
    centroids = np.mean(vertices[triangles], axis=1)

    # Perform Ward's hierarchical clustering
    num_clusters = len(vertices) // lambda_val

    # Apply Ward's method on centroids
    Z = ward(centroids)
    labels = fcluster(Z, t=num_clusters, criterion='maxclust')
    labels = labels - 1 

    # Create a one-hot encoding matrix using numpy advanced indexing
    one_hot_matrix = np.zeros((len(labels), num_clusters), dtype=int)
    one_hot_matrix[np.arange(len(labels)), labels] = 1
    return one_hot_matrix, labels

def get_clustering_matrix(mesh, labels):
    triangles = np.asarray(mesh.triangles)
    num_faces = len(triangles)
    
    C_matrix = np.zeros((num_faces, num_faces), dtype=int)

    # Fill the matrix such that Cij = 1 if Fi and Fj belong to the same cluster
    for i in range(num_faces):
        for j in range(num_faces):
            if labels[i] == labels[j]:
                C_matrix[i, j] = 1
    
    return C_matrix

def get_laplacian_features(mesh, target_faces, E):
    adj_matrix, degee_matrix = get_mesh_adjacency_and_degree_matrices(mesh, target_faces)
    eigenvectors, _ = get_eigenvectors(adj_matrix, degee_matrix)

    return eigenvectors[:, :E]

def compute_face_features(mesh, target_faces=2412, E=20):
    # Extract vertex coordinates and face indices from the mesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Compute triangle normals
    normals = np.asarray(compute_triangle_normals(mesh))
    pos_encoding = get_laplacian_features(mesh, target_faces, E)
    
    # Initialize the feature matrix
    num_faces = faces.shape[0]
    E = pos_encoding.shape[1]  # Number of positional encoding features
    feature_matrix = np.zeros((num_faces, 9 + 3 + E))  # 3 vertices * 3 coordinates + 3 normal + E positional encodings

    # Extract vertex coordinates for each face
    vertex_features = vertices[faces]  # Shape (F, 3, 3)
    
    # Flatten the vertex coordinates and place them in the feature matrix
    feature_matrix[:, :9] = vertex_features.reshape(num_faces, -1)  # Reshape to (F, 9)

    # Insert normal vectors
    feature_matrix[:, 9:12] = normals
    
    # Insert positional encoding features
    feature_matrix[:, 12:] = pos_encoding
    
    return feature_matrix

if __name__ == "__main__":
    mesh = load_mesh("simplified_mesh.obj")
    cluster_features, labels = get_clustering_features(mesh)
    visualize_positional_encoding(mesh, get_laplacian_features(mesh, 2412, 10))
    visualize_clustering_feature(mesh, cluster_features)