import numpy as np
from scipy.cluster.hierarchy import ward, fcluster
from utils import load_mesh, get_mesh_adjacency_and_degree_matrices, get_eigenvectors, compute_triangle_normals
from visualization import visualize_clustering_feature, visualize_positional_encoding
import torch
import random

def get_clustering_features(mesh, vertices_max=1200, lambda_val=8):
    # Convert Open3D mesh to NumPy arrays
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Compute features for each triangle (centroid in this case)
    centroids = np.mean(vertices[triangles], axis=1)

    # Perform Ward's hierarchical clustering
    num_clusters = vertices_max // lambda_val

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

def get_laplacian_features(mesh, E):
    adj_matrix, degree_matrix = get_mesh_adjacency_and_degree_matrices(mesh)
    eigenvectors, _ = get_eigenvectors(adj_matrix, degree_matrix)

    return eigenvectors[:, :E+1]

def compute_face_features(mesh, E=20):
    # Extract vertex coordinates and face indices from the mesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Compute triangle normals
    normals = np.asarray(compute_triangle_normals(mesh))
    pos_encoding = get_laplacian_features(mesh, E)
    
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

def augment_features(T_features, faces_area, translation_range=0.1, rotation_range=30, scale_range=(0.8, 1.2)):
    """
    Augment face features with random translation, rotation, and scaling, followed by normalization.

    Args:
    - T_features (torch.Tensor): Tensor of face features (num_faces, feature_dim).
    - faces_area (torch.Tensor): Tensor containing the areas of the faces (num_faces).
    - translation_range (float): Max translation value for positional components.
    - rotation_range (float): Max rotation in degrees.
    - scale_range (tuple): Min and max scaling factors.
    
    Returns:
    - augmented_features (torch.Tensor): Augmented and normalized face features.
    - augmented_faces_area (torch.Tensor): Augmented and scaled face areas.
    """
    
    num_faces = T_features.shape[0]
    
    # Extract vertex coordinates (first 9 columns of T_features)
    vertex_coords = T_features[:, :9].reshape(num_faces, 3, 3)  # Shape: (num_faces, 3, 3)
    
    # Extract normal vectors (columns 9:12)
    normals = T_features[:, 9:12]

    # 1. Apply random translation
    translation = (torch.rand(3) - 0.5) * 2 * translation_range  # Random translation vector
    vertex_coords += translation  # Translate vertex coordinates

    # 2. Apply random rotation
    angle = random.uniform(-rotation_range, rotation_range) * torch.pi / 180  # Convert to radians
    cos, sin = torch.cos(angle), torch.sin(angle)
    
    # Random 3D rotation matrix around z-axis (can be extended to arbitrary axes)
    rotation_matrix = torch.tensor([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
    ])

    # Rotate vertex coordinates and normals
    vertex_coords = torch.matmul(vertex_coords, rotation_matrix)
    normals = torch.matmul(normals, rotation_matrix)

    # 3. Apply random scaling
    scale_factor = random.uniform(*scale_range)
    vertex_coords *= scale_factor  # Scale vertex coordinates
    faces_area *= scale_factor ** 2  # Scaling affects areas quadratically

    # Normalize vertex coordinates to [-1, 1]
    min_vals, _ = vertex_coords.min(dim=0, keepdim=True)
    max_vals, _ = vertex_coords.max(dim=0, keepdim=True)
    vertex_coords = 2 * (vertex_coords - min_vals) / (max_vals - min_vals + 1e-7) - 1  # Min-max normalization

    # Normalize normals (if needed, though normals are unit vectors, normalization may not be necessary)
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-7)  # Normalize normals to unit vectors

    # Update the augmented features
    augmented_features = T_features.clone()
    augmented_features[:, :9] = vertex_coords.reshape(num_faces, -1)  # Update vertex positions
    augmented_features[:, 9:12] = normals  # Update normals

    return augmented_features, faces_area

if __name__ == "__main__":
    mesh = load_mesh("dataset/raw_meshes/7.off")
    adj_matrix, deg_matrix = get_mesh_adjacency_and_degree_matrices(mesh)
    cluster_features, labels = get_clustering_features(mesh)
    visualize_positional_encoding(mesh, get_laplacian_features(mesh, 0))
    visualize_clustering_feature(mesh, cluster_features)