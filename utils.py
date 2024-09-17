import open3d as o3d
import numpy as np
from scipy.sparse import lil_matrix, diags
from scipy.linalg import eigh

def load_mesh(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    return mesh

def compute_triangle_normals(mesh):
    mesh.compute_triangle_normals()
    return mesh.triangle_normals  

def get_mesh_adjacency_and_degree_matrices(mesh, target_size):
    
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

    # If the target size is larger than the number of triangles, pad the matrices
    if num_triangles < target_size:
        padded_adj_matrix = np.zeros((target_size, target_size), dtype=int)
        padded_adj_matrix[:num_triangles, :num_triangles] = adj_matrix
        
        padded_degree_matrix = np.zeros((target_size, target_size), dtype=int)
        padded_degree_matrix[:num_triangles, :num_triangles] = degree_matrix
        
        return padded_adj_matrix, padded_degree_matrix
    
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
    
def log_transform(matrix):
    log_matrix = np.where(matrix == 0, -np.inf, np.log(matrix))
    
    return log_matrix