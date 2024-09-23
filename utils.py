import open3d as o3d
import numpy as np
from scipy.sparse import lil_matrix, diags
from scipy.linalg import eigh
from collections import Counter
from scipy.spatial import KDTree
import torch

def load_mesh(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    return mesh

def convert_to_standard_mesh(t_mesh):
    # Create a new standard TriangleMesh object
    standard_mesh = o3d.geometry.TriangleMesh()

    # Extract vertices and triangles from the open3d.t.geometry.TriangleMesh
    vertices = t_mesh.vertex.positions.numpy()  # Convert tensor to numpy array
    triangles = t_mesh.triangle.indices.numpy()  # Convert tensor to numpy array
    
    # Set vertices and triangles to the new standard TriangleMesh
    standard_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    standard_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Optionally, you can copy other attributes such as vertex normals or colors
    if 'normals' in t_mesh.vertex:
        vertex_normals = t_mesh.vertex.normals.numpy()
        standard_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
        
    if 'normals' in t_mesh.triangle:
        triangle_normals = t_mesh.triangle.normals.numpy()
        standard_mesh.triangle_normals = o3d.utility.Vector3dVector(triangle_normals)
    
    return standard_mesh

def compute_triangle_normals(mesh):
    mesh.compute_triangle_normals()
    return mesh.triangle_normals  

def pad_np_matrix(matrix, target_size, pad_columns=False):
    F, N = matrix.shape
    
    if F < target_size:
        if pad_columns == False:
            # Pad rows to target_size, keep N
            padded_matrix = np.zeros((target_size, N), dtype=matrix.dtype)
            padded_matrix[:F, :] = matrix
        else:
            # Pad rows and columns to target_size
            padded_matrix = np.zeros((target_size, target_size), dtype=matrix.dtype)
            padded_matrix[:F, :N] = matrix
            
        return padded_matrix
    else:
        return matrix

def get_mesh_adjacency_and_degree_matrices(mesh):
    
    # Extract triangles from the Open3D mesh
    triangles = np.asarray(mesh.triangles)
    num_triangles = len(triangles)
    
    # Create a sparse adjacency matrix
    adj_matrix = lil_matrix((num_triangles, num_triangles), dtype=int)
    
    # Create a mapping of edges to the triangles they belong to
    edge_to_triangles = {}
    
    # Build a map of each edge to the triangles that share it
    for i, tri in enumerate(triangles):
        # Each triangle has 3 edges
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for edge in edges:
            # Store the edge in a sorted manner so that (i, j) and (j, i) are treated the same
            edge = tuple(sorted(edge))
            if edge in edge_to_triangles:
                edge_to_triangles[edge].append(i)
            else:
                edge_to_triangles[edge] = [i]
    
    # Fill the adjacency matrix by marking adjacent triangles
    for edge, adj_tris in edge_to_triangles.items():
        if len(adj_tris) == 2:  # Only if two triangles share an edge
            t1, t2 = adj_tris
            adj_matrix[t1, t2] = 1
            adj_matrix[t2, t1] = 1

    # Convert the adjacency matrix to dense for padding, if necessary
    adj_matrix = adj_matrix.toarray()
    
    # Compute the degree matrix (sum of rows for each triangle)
    degrees = np.sum(adj_matrix, axis=1)
    degree_matrix = diags(degrees, format='csr').toarray()
    
    return adj_matrix, degree_matrix

def get_eigenvectors(adj_matrix, degree_matrix):
    degree_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    
    normalized_laplacian = np.eye(degree_matrix.shape[0]) - degree_inv_sqrt @ adj_matrix @ degree_inv_sqrt
    eigenvalues, eigenvectors = eigh(normalized_laplacian)
    
    non_zero_indices = np.where(np.abs(eigenvalues) > 1e-10)[0]
    non_trivial_eigenvalues = eigenvalues[non_zero_indices]
    non_trivial_eigenvectors = eigenvectors[:, non_zero_indices]

    return non_trivial_eigenvectors, non_trivial_eigenvalues

def export_mesh(mesh, filename):
    o3d.io.write_triangle_mesh(filename, mesh)
    
def log_transform_np(matrix):
    log_matrix = np.where(matrix == 0, -np.inf, np.log(matrix))
    
    return log_matrix

def log_transform_tensor(matrix):
    # Replace zeros with -inf, ensure no zero or negative values for log calculation
    mask = torch.where(matrix == 0, torch.tensor(float('-inf')), torch.tensor(0.0))  # Creates a mask with -inf and 0
    return mask

def map_labels_to_new_mesh(original_mesh, original_labels, new_mesh):
    """
    Map labels from the original mesh to the new mesh and smooth labels for isolated faces.
    
    Args:
        original_mesh (o3d.geometry.TriangleMesh): The original mesh with labels.
        original_labels (np.ndarray): The labels of the original mesh faces.
        new_mesh (o3d.geometry.TriangleMesh): The new mesh that needs updated labels.

    Returns:
        np.ndarray: The updated labels for the new mesh.
    """
    # Convert meshes to numpy arrays for face indices
    original_faces = np.asarray(original_mesh.triangles)
    new_faces = np.asarray(new_mesh.triangles)
    original_vertices = np.asarray(original_mesh.vertices)
    new_vertices = np.asarray(new_mesh.vertices)

    # Calculate centroids of faces
    def face_centroids(faces, vertices):
        return np.mean(vertices[faces], axis=1)

    original_centroids = face_centroids(original_faces, original_vertices)
    new_centroids = face_centroids(new_faces, new_vertices)
    
    # Build a KDTree for fast nearest neighbor search
    tree = KDTree(new_centroids)
    _, indices = tree.query(original_centroids, k=1)
    
    # Map original faces to new faces based on closest centroids
    face_to_new_face = indices.flatten()
    updated_labels = np.full(len(new_faces), -1, dtype=int)  # -1 indicates no label
    
    # Map labels from original mesh to new mesh
    for i, new_face_index in enumerate(face_to_new_face):
        if i < len(original_labels):  # Ensure we do not exceed original labels
            updated_labels[new_face_index] = original_labels[i]
    
    # Build adjacency list for faces in new mesh
    def build_adjacency_list(faces):
        tree = KDTree(np.mean(new_vertices[faces], axis=1))
        adjacency_list = []
        for face in faces:
            _, indices = tree.query(np.mean(new_vertices[face], axis=0), k=7)  # Find 4 nearest neighbors
            adjacency_list.append(indices.tolist())
        return adjacency_list
    
    adjacency_list = build_adjacency_list(new_faces)
    
    # Smooth labels for isolated faces with fallback to more neighbors
    def smooth_labels(labels, adjacency_list, max_iterations=10):
        for _ in range(max_iterations):
            new_labels = labels.copy()
            for i, label in enumerate(labels):
                if label == -1:  # Isolated face
                    # Check neighbors first
                    neighbor_labels = [labels[j] for j in adjacency_list[i] if labels[j] != -1]
                    
                    if not neighbor_labels:
                        # If no valid neighbor labels, look at second-level neighbors
                        second_neighbors = set()
                        for j in adjacency_list[i]:
                            second_neighbors.update(adjacency_list[j])
                        neighbor_labels = [labels[k] for k in second_neighbors if labels[k] != -1]
                    
                    if neighbor_labels:
                        # Only consider valid labels from the original set
                        valid_labels = set(original_labels)
                        neighbor_labels = [lbl for lbl in neighbor_labels if lbl in valid_labels]
                        if neighbor_labels:
                            most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
                            new_labels[i] = most_common_label
            
            # Check if labels have changed; if not, exit early
            if np.array_equal(labels, new_labels):
                break
            
            labels = new_labels
        
        return labels

    updated_labels = smooth_labels(updated_labels, adjacency_list)

    # Final pass to fill any remaining -1 labels
    for i in range(len(updated_labels)):
        if updated_labels[i] == -1:
            # Consider all original labels if still -1
            neighbor_labels = [updated_labels[j] for j in adjacency_list[i] if updated_labels[j] != -1]
            valid_labels = set(original_labels)
            neighbor_labels = [lbl for lbl in neighbor_labels if lbl in valid_labels]
            if neighbor_labels:
                most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
                updated_labels[i] = most_common_label
            else:
                # Fallback: Assign a label from the original set if all else fails
                updated_labels[i] = original_labels[0] if original_labels.size > 0 else -1  # or some default label

    return updated_labels

def compute_triangle_areas(mesh):
    """
    Compute the area of each triangle in the mesh.

    Parameters:
    - mesh (o3d.geometry.TriangleMesh): The input Open3D mesh.

    Returns:
    - np.ndarray: Array of areas for each triangle in the mesh.
    """
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise TypeError("Input must be an Open3D TriangleMesh object.")
    
    # Ensure the mesh has vertices and triangles
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError("The mesh does not contain any vertices or triangles.")
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    def triangle_area(v0, v1, v2):
        # Compute the area of a triangle given its three vertices
        a = np.linalg.norm(v1 - v0)
        b = np.linalg.norm(v2 - v1)
        c = np.linalg.norm(v0 - v2)
        s = (a + b + c) / 2.0
        return np.sqrt(s * (s - a) * (s - b) * (s - c))

    areas = []
    for tri in triangles:
        v0, v1, v2 = vertices[tri]
        area = triangle_area(v0, v1, v2)
        areas.append(area)

    return np.array(areas)