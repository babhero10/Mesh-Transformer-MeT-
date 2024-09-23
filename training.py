# Training, Evaluation, and Testing loops
import torch
from torch_geometric.loader import DataLoader
from MeshDataloader_coseg import MeshDataset
from Model_blocks import MeshTransformer
from features_extraction import augment_features
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from visualization import visualize_mesh_pyvista
from utils import load_mesh
from mesh_preprocessing import preprocess_mesh

def calculate_accuracy(predictions, targets, triangle_areas):
    # Assumes predictions and targets are numpy arrays of shape (num_triangles,)
    correct = predictions == targets
    correct_areas = triangle_areas[correct]
    total_area = triangle_areas.sum()
    correct_area = correct_areas.sum()
    return correct_area / total_area
    # return len(triangle_areas[correct]) / len(targets)

def weighted_ce_loss(input, target, weights):
    # Normalize weights to have mean 1
    normalized_weights = weights / weights.mean()

    # Compute the standard cross-entropy loss (without reduction)
    loss = torch.nn.functional.cross_entropy(input, target, reduction='none')
    
    # Multiply by the normalized weights
    weighted_loss = loss * normalized_weights
    
    # Return the mean of the weighted losses
    return weighted_loss.mean()

def iou(pred, target, num_classes):
    """
    Calculate Intersection over Union (IoU) for each class.

    Args:
        pred (torch.Tensor): Predictions from the model, of shape (N,).
        target (torch.Tensor): Ground truth labels, of shape (N,).
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: IoU for each class.
    """
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious.append(torch.tensor(float('nan')))
        else:
            ious.append(intersection / union)
    return torch.tensor(ious)

# Training Loop
def train(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, weight_decay=0.01, device='cuda'):
    # Initialize loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # train_losses, val_losses = [], []
    # train_accuracies, val_accuracies = [], []
    train_losses = []
    train_accuracies = []
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_epochs):
        model.train()
        if epoch == 500:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate/10, weight_decay=weight_decay)
            torch.save(model.state_dict(), f'models/model_epoch_{epoch + 1}.pth')
        if epoch == 750:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate/2, weight_decay=weight_decay)
            torch.save(model.state_dict(), f'models/model_epoch_{epoch + 1}.pth')
            
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        triangle_areas_sum = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            T_features, J_features, A_matrix, C_matrix, faces_area, y = [x.to(device) for x in batch]
            C_matrix = C_matrix.float()

            # Check for NaNs in input data
            if torch.isnan(T_features).any() or torch.isnan(J_features).any():
                print("NaNs found in input features!")

            # Forward pass
            S_out, P_out = model(T_features, J_features, A_matrix, C_matrix)
            S_out = S_out.view(-1, 4)

            # Check for NaNs in S_out
            if torch.isnan(S_out).any():
                print("NaNs found in S_out!")

            # Adjust labels for zero-based indexing, ignoring zeros
            y_adjusted = y.clone()
            y_adjusted[y != 0] -= 1
            y_adjusted = y_adjusted.view(-1)

            faces_area = faces_area.view(-1)

            # faces_area = torch.log1p(faces_area)
            faces_area = faces_area*100

            # Calculate loss
            loss = weighted_ce_loss(S_out, y_adjusted, faces_area)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            
            predictions = torch.argmax(S_out, dim=1)
            targets = y_adjusted
            
            accuracy = calculate_accuracy(predictions, targets, faces_area)
            epoch_accuracy += accuracy * faces_area.sum()
            triangle_areas_sum += faces_area.sum()
        # monitor_gradients(model)

        # Average loss and accuracy for the epoch
        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(epoch_accuracy / triangle_areas_sum)
        
        # Validation
        # val_loss, val_accuracy = evaluate(model, val_loader, device)
        # val_losses.append(val_loss)
        # val_accuracies.append(val_accuracy)
        
        # Save model state
        # torch.save(model.state_dict(), f'models/model_epoch_{epoch + 1}.pth')
        
        # Print epoch metrics
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")

    torch.save(model.state_dict(), f'models/model_epoch_{epoch + 1}.pth')
    
    # Plot loss and accuracy graphs
    # plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

# Evaluation Loop
def evaluate(model, loader, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_triangle_areas = 0.0
    
    with torch.no_grad():
        for batch in loader:
            T_features, J_features, A_matrix, C_matrix, faces_area, y = [x.to(device) for x in batch]
            
            # Forward pass
            S_out, _ = model(T_features, J_features, A_matrix, C_matrix)
            
            # Calculate loss
            loss = weighted_ce_loss(S_out, y, faces_area)
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(S_out, dim=1).cpu().numpy()
            targets = y.cpu().numpy()
            areas = faces_area.cpu().numpy()
            accuracy = calculate_accuracy(predictions, targets, areas)
            total_accuracy += accuracy * areas.sum()
            total_triangle_areas += areas.sum()
    
    avg_loss = total_loss / len(loader)
    avg_accuracy = total_accuracy / total_triangle_areas
    return avg_loss, avg_accuracy

# Plotting metrics
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')
    
    plt.tight_layout()
    plt.show()
    
def test(model, test_loader, device='cuda'):
    model.eval()
    predicted_labels = []
    true_labels = []
    total_accuracy = 0.0
    total_triangle_areas = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            T_features, J_features, A_matrix, C_matrix, faces_area, y = [x.to(device) for x in batch]
            
            # Forward pass
            S_out, _ = model(T_features, J_features, A_matrix, C_matrix)
            
            # Get predicted labels
            predictions = torch.argmax(S_out, dim=1).cpu().numpy()
            predicted_labels.extend(predictions)
            true_labels.extend(y.cpu().numpy())
            
            # Calculate accuracy
            targets = y.cpu().numpy()
            areas = faces_area.cpu().numpy()
            accuracy = calculate_accuracy(predictions, targets, areas)
            total_accuracy += accuracy * areas.sum()
            total_triangle_areas += areas.sum()
    
    avg_accuracy = total_accuracy / total_triangle_areas
    return predicted_labels, true_labels, avg_accuracy

def monitor_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Layer: {name}, Gradient Norm: {param.grad.norm()}")
        else:
            print(f"Layer: {name}, No gradient computed, Freeze: {not param.requires_grad}")
      

def get_and_visualize_outputs(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    i = 0
    text = ['dataset/raw_meshes/0.off', 'dataset/raw_meshes/1.off', 'dataset/raw_meshes/10.off']
    with torch.no_grad():  # Disable gradient calculation
        for data in dataloader:
            T_features, J_features, A_matrix, C_matrix, faces_area, y = [x.to(device) for x in data]
            C_matrix = C_matrix.float()

            # Check for NaNs in input data
            if torch.isnan(T_features).any() or torch.isnan(J_features).any():
                print("NaNs found in input features!")

            # Forward pass
            S_out, P_out = model(T_features, J_features, A_matrix, C_matrix)
            S_out = S_out.view(-1, 4)
            S_out = torch.argmax(S_out, dim=1)
            mesh = preprocess_mesh(load_mesh(text[i]), 1200)
            print(S_out.cpu()[:len(mesh.triangles)].max(), S_out.cpu()[:len(mesh.triangles)].min())
            # Visualize the output
            visualize_mesh_pyvista(mesh, S_out.cpu()[:len(mesh.triangles)])  # Move output to CPU if needed
            i += 1
      
def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = MeshDataset(root='dataset', raw_mesh_folder='raw_meshes', seg_folder='seg')
    
    # Split into training and validation datasets
    train_size = int(0.01 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
   
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = train_loader
    # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    model = MeshTransformer(
        t_dim=27,
        J_dim=2412,
        P_dim=512,
        E_dim=1024,
        num_of_labels=4,
        num_of_encoder_layers=3,
        dropout=0.0,
        num_heads=8,
        add_norm=True,
        residual=True
    ).to(device)

    # Train the model
    # num_epochs = 1000
    # learning_rate = 5e-6
    # weight_decay = 0
    # train(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay=weight_decay, device=device)
    model.load_state_dict(torch.load('models/model_epoch_1000.pth'))
    get_and_visualize_outputs(model, train_loader, device)
    # # Evaluate the model on the validation set
    # val_loss, val_accuracy = evaluate(model, val_loader, device)
    # print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    # # Test the model
    # test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  # Assuming val_loader as test loader
    # predicted_labels, true_labels, test_accuracy = test(model, test_loader, device)
    # print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # # Optionally, save the predicted labels to a file
    # with open('predicted_labels.txt', 'w') as f:
    #     for pred, true in zip(predicted_labels, true_labels):
    #         f.write(f"Predicted: {pred}, True: {true}\n")
    
    # print("Testing complete and labels saved.")

# Run the main code
if __name__ == "__main__":
    main()
    
    