import os
import requests
import zipfile
from sklearn.decomposition import PCA
from tqdm import tqdm
from torch_geometric.data import Dataset, Data
import torch
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import shutil

class MeshDataset(Dataset):
    SHAPE_URL = "https://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Chairs/shapes.zip"
    LABEL_URL = "https://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Chairs/gt.zip"
    
    def __init__(self, root, raw_mesh_folder, seg_folder, transform=None, pre_transform=None, pre_filter=None):
        self.raw_mesh_folder = os.path.join(root, raw_mesh_folder)
        self.seg_folder = os.path.join(root, seg_folder)
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

    def visualize_colored_mesh(self, mesh_file, label_file):
        """
        Visualizes the mesh with colors according to the labels.

        Args:
            mesh_file (str): Path to the .off mesh file.
            label_file (str): Path to the label file.
        """
        # Load the mesh
        mesh = trimesh.load(mesh_file)

        # Load the labels
        labels = self.read_seg_file(label_file)

        # Ensure the labels match the number of faces
        if len(labels) != len(mesh.faces):
            raise ValueError("The number of labels does not match the number of faces in the mesh.")

        # Assign a color to each unique label
        unique_labels = np.unique(labels)
        color_map = plt.get_cmap('jet', len(unique_labels))
        
        # Generate colors
        color_dict = {label: color_map(i) for i, label in enumerate(unique_labels)}
        face_colors = np.array([color_dict[label] for label in labels])

        # Set face colors
        mesh.visual.face_colors = (face_colors[:, :3] * 255).astype(np.uint8)

        # Show the mesh
        mesh.show()
    
    def process(self):
        # Iterate over the raw files and process them
        idx = 0
        raw_mesh_files = sorted(os.listdir(self.raw_mesh_folder))
        seg_files = sorted(os.listdir(self.seg_folder))

        for mesh_file, seg_file in zip(raw_mesh_files, seg_files):
            mesh_path = os.path.join(self.raw_mesh_folder, mesh_file)
            seg_path = os.path.join(self.seg_folder, seg_file)

            # Load the raw mesh
            mesh = trimesh.load(mesh_path)
            vertices = mesh.vertices
            normals = mesh.vertex_normals
            edges = np.array(mesh.edges_unique)

            # Load the segmentation labels (face labels)
            face_labels = self.read_seg_file(seg_path)

            # Convert face labels to vertex labels (optional)
            vertex_labels = self.convert_face_to_vertex_labels(mesh, face_labels)
            
            # Combine PCA coordinates and normal vectors as features
            pca_coords = self.get_pca_coordinates(mesh)
            combined_features = np.hstack((pca_coords, normals))
            x = torch.tensor(combined_features, dtype=torch.float)

            edge_index = torch.tensor(edges.T, dtype=torch.long)  # Transpose to match PyG format
            y = torch.tensor(vertex_labels, dtype=torch.long)
            y = y - 1
            
            # Create the graph data object
            data = Data(x=x, edge_index=edge_index, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def convert_face_to_vertex_labels(self, mesh, face_labels):
        """
        Converts face labels to vertex labels using majority voting.

        Args:
            mesh (trimesh.Trimesh): The mesh object.
            face_labels (np.ndarray): Array of face labels.

        Returns:
            np.ndarray: Array of vertex labels.
        """
        num_vertices = len(mesh.vertices)
        vertex_labels = np.zeros(num_vertices, dtype=int)
        vertex_face_counts = np.zeros(num_vertices, dtype=int)

        for face_id, label in enumerate(face_labels):
            for vertex_id in mesh.faces[face_id]:
                vertex_labels[vertex_id] += label
                vertex_face_counts[vertex_id] += 1

        # Compute majority label for each vertex
        vertex_labels = np.round(vertex_labels / vertex_face_counts).astype(int)
        return vertex_labels

    def get_pca_coordinates(self, mesh):
        """
        Perform PCA on the mesh vertices.

        Args:
            mesh (trimesh.Trimesh): mesh object from trimesh.
        
        Returns:
            np.ndarray: PCA coordinates.
        """
        pca = PCA(n_components=3)
        pca_coords = pca.fit_transform(mesh.vertices)
        return pca_coords

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
        return np.array(labels, dtype=np.int64)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))


# dataset = MeshDataset(root='dataset', raw_mesh_folder='raw_meshes', seg_folder='seg')
# dataset.download()

# mesh_file = 'dataset/raw_meshes/6.off'
# label_file = 'dataset/seg/6.seg'
# dataset.visualize_colored_mesh(mesh_file, label_file)