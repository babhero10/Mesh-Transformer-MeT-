import os
import requests
import zipfile
from tqdm import tqdm
from torch_geometric.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import shutil
from utils import load_mesh, get_mesh_adjacency_and_degree_matrices, pad_np_matrix, map_labels_to_new_mesh, compute_triangle_areas
from features_extraction import compute_face_features, get_clustering_matrix, get_clustering_features
from mesh_preprocessing import preprocess_mesh
from visualization import visualize_mesh_pyvista
from tqdm import tqdm

class MeshDataset(Dataset):
    SHAPE_URL = "https://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Vases/shapes.zip"
    LABEL_URL = "https://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Vases/gt.zip"
    
    def __init__(self, root, raw_mesh_folder, seg_folder, target_max_vertices=1200, target_faces=2412, num_of_eigenvectors=15, lambda_val=8, transform=None, pre_transform=None, pre_filter=None):
        self.raw_mesh_folder = os.path.join(root, raw_mesh_folder)
        self.seg_folder = os.path.join(root, seg_folder)
        self.target_faces = target_faces
        self.target_max_vertices = target_max_vertices
        self.num_of_eigenvectors = num_of_eigenvectors
        self.lambda_val = lambda_val
        
        super(MeshDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return sorted(os.listdir(self.raw_mesh_folder))

    @property
    def processed_file_names(self):
        return [f'data_{idx}.pt' for idx in range(len(self.raw_file_names))]

    def download(self):
        if os.path.exists(self.raw_mesh_folder) and os.listdir(self.raw_mesh_folder) \
                and os.path.exists(self.seg_folder) and os.listdir(self.seg_folder):
            print("Dataset already downloaded and extracted.")
            return
        
        # Download shape files
        print(f"Downloading {self.SHAPE_URL}...")
        shape_zip_path = os.path.join(self.root, "shapes.zip")
        self.download_url(self.SHAPE_URL, shape_zip_path)
        
        # Download label files
        print(f"Downloading {self.LABEL_URL}...")
        label_zip_path = os.path.join(self.root, "gt.zip")
        self.download_url(self.LABEL_URL, label_zip_path)
        
        # Extract shape files
        print("Extracting shape files...")
        with zipfile.ZipFile(shape_zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_mesh_folder)

        # Extract label files
        print("Extracting label files...")
        with zipfile.ZipFile(label_zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.seg_folder)

        # Move files from the inner folders to the specified folders
        self.move_files_to_target(self.raw_mesh_folder, "shapes")
        self.move_files_to_target(self.seg_folder, "gt")

        # Remove the zip files after extraction
        os.remove(shape_zip_path)
        os.remove(label_zip_path)

        print("Download, extraction, and cleanup complete.")
    
    def move_files_to_target(self, target_folder, inner_folder_name):
        """
        Move files from the extracted inner folder to the target folder.

        Args:
            target_folder (str): The folder where files should be moved.
            inner_folder_name (str): The name of the inner folder inside the .zip archive.
        """
        inner_folder_path = os.path.join(target_folder, inner_folder_name)
        
        # Move all files from the inner folder to the target folder
        for filename in os.listdir(inner_folder_path):
            shutil.move(os.path.join(inner_folder_path, filename), target_folder)
        
        # Remove the now-empty inner folder
        os.rmdir(inner_folder_path)

    def download_url(self, url, save_path):
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))

        with open(save_path, 'wb') as file, tqdm(
            desc=save_path,
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))
    
    def visualize_colored_mesh(self, mesh, labels):
        """
        Visualizes the mesh with colors according to the labels.

        Args:
            mesh_file (str): Path to the .off mesh file.
            label_file (str): Path to the label file.
        """
        # Ensure the labels match the number of faces
        if labels.shape[0] != len(mesh.triangles):
            raise ValueError("The number of labels does not match the number of faces in the mesh.")

        # Assign a color to each unique label
        unique_labels = np.unique(labels)
        color_map = plt.get_cmap('jet', len(unique_labels))  # Choose a color map
        
        # Generate colors for each label
        color_dict = {label: color_map(i)[:3] for i, label in enumerate(unique_labels)}  # Ignore alpha channel
        face_colors = np.array([color_dict[label] for label in labels])

        # Convert face colors to the required format for PyVista
        face_colors_rgb = (face_colors * 255).astype(np.uint8)  # Scale to [0, 255] for RGB

        # Visualize the mesh using PyVista
        visualize_mesh_pyvista(mesh, face_colors_rgb)
    
    def process(self):
        # Iterate over the raw files and process them
        idx = 0
        raw_mesh_files = sorted(os.listdir(self.raw_mesh_folder))
        seg_files = sorted(os.listdir(self.seg_folder))

        # Wrap the zip of raw_mesh_files and seg_files with tqdm for progress tracking
        for mesh_file, seg_file in tqdm(zip(raw_mesh_files, seg_files), total=len(raw_mesh_files), desc="Processing Meshes"):
            mesh_path = os.path.join(self.raw_mesh_folder, mesh_file)
            seg_path = os.path.join(self.seg_folder, seg_file)
            
            # Load the raw mesh
            mesh = load_mesh(mesh_path)

            # Load the segmentation labels (face labels)
            face_labels = self.read_seg_file(seg_path)
            preprocessed_mesh = preprocess_mesh(mesh, target_max_vertices=self.target_max_vertices)
            face_labels = map_labels_to_new_mesh(mesh, face_labels, preprocessed_mesh)
            faces_area = compute_triangle_areas(preprocessed_mesh)
            
            if face_labels.min() < 1:
                print("Mesh: ", mesh_file)
                print(f"Warning: face labels min is {face_labels.min()}, but expected {1}")
            if face_labels.max() > 4:
                print("Mesh: ",  mesh_file)
                print(f"Warning: face labels max is {face_labels.max()}, but expected {4}")
            
            if face_labels.shape[0] < self.target_faces:
                face_labels = np.pad(face_labels, (0, self.target_faces - face_labels.shape[0]), mode='constant')
                faces_area = np.pad(faces_area, (0, self.target_faces - faces_area.shape[0]), mode='constant')

            # Compute face features using the provided function
            T_features = compute_face_features(preprocessed_mesh, E=self.num_of_eigenvectors)
            J_features, labels = get_clustering_features(preprocessed_mesh, vertices_max=self.target_max_vertices, lambda_val=self.lambda_val)
            A_matrix, _ = get_mesh_adjacency_and_degree_matrices(preprocessed_mesh)
            C_matrix = get_clustering_matrix(preprocessed_mesh, labels)

            A_matrix = pad_np_matrix(A_matrix, self.target_faces, pad_columns=True)
            C_matrix = pad_np_matrix(C_matrix, self.target_faces, pad_columns=True)
            T_features = pad_np_matrix(T_features, self.target_faces)
            J_features = pad_np_matrix(J_features, self.target_faces)

            # Convert to tensors
            T_features = torch.tensor(T_features, dtype=torch.float)
            J_features = torch.tensor(J_features, dtype=torch.long)
            A_matrix = torch.tensor(A_matrix, dtype=torch.uint8)
            C_matrix = torch.tensor(C_matrix, dtype=torch.float)
            
            # Define expected shapes
            expected_shapes = {
                'T_features': (self.target_faces, 27),
                'J_features': (self.target_faces, self.target_max_vertices // self.lambda_val),
                'A_matrix': (self.target_faces, self.target_faces),
                'C_matrix': (self.target_faces, self.target_faces),
                'Labels': (self.target_faces, ),
                'Faces_area': (self.target_faces, )
            }
            
            # Map local variables to their names
            tensors = {
                'T_features': T_features,
                'J_features': J_features,
                'A_matrix': A_matrix,
                'C_matrix': C_matrix,
                'Labels': face_labels,
                'Faces_area': faces_area,
            }

            # Check shapes and print messages if there are issues
            for name, tensor in tensors.items():
                if tensor.shape != torch.Size(expected_shapes[name]):
                    print(f"Warning: {name} shape is {tensor.shape}, but expected {expected_shapes[name]}")
                
            
            y = torch.tensor(face_labels, dtype=torch.long)
            faces_area = torch.tensor(faces_area, dtype=torch.float32)
            
            # Save the features and labels
            torch.save((T_features, J_features, A_matrix, C_matrix, faces_area, y), os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def read_seg_file(self, seg_file):
        """
        Read the segmentation file and return the labels.

        Args: 
            seg_file (str): Path to the segmentation file.

        Returns:
            np.ndarray: Labels as a numpy array.
        """
        with open(seg_file, 'r') as f:
            labels = [int(line.strip()) for line in f]
        return np.array(labels, dtype=np.int8)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))


if __name__ == "__main__":
    dataset = MeshDataset(root='dataset', raw_mesh_folder='raw_meshes', seg_folder='seg')
    dataset.download()

    mesh = load_mesh('dataset/raw_meshes/6.off')
    label = dataset.read_seg_file('dataset/seg/6.seg')
    dataset.visualize_colored_mesh(mesh, label)