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
from visualization import visualize_colored_mesh
from utils import load_mesh
from mesh_preprocessing import preprocess_mesh
from torch.amp import GradScaler, autocast

def calculate_accuracy(predictions, targets, triangle_areas):
    correct = predictions == targets
    correct_areas = triangle_areas[correct]
    total_area = triangle_areas.sum()
    correct_area = correct_areas.sum()
    return correct_area / total_area

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
def train(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, weight_decay=0.01, device='cuda', accumulation_steps=2):
    # Initialize optimizer and mixed precision scaler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler(device='cuda')  # For mixed precision training

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Reset gradients at the start of each epoch

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
            T_features, J_features, A_matrix, C_matrix, faces_area, y = [x.to(device) for x in batch[:-1]]
            T_features, faces_area = augment_features(T_features, faces_area, rotation_range=180)

            if torch.isnan(T_features).any() or torch.isnan(J_features).any():
                print("NaNs found in input features!")

            # Mixed precision forward pass
            with autocast(device_type='cuda'):  
                S_out, P_out = model(T_features, J_features, A_matrix, C_matrix)
                S_out = S_out.view(-1, 4)

                if torch.isnan(S_out).any():
                    print("NaNs found in S_out!")

                y_adjusted = y.clone()
                y_adjusted[y != 0] -= 1  # Adjust labels for zero-based indexing
                y_adjusted = y_adjusted.view(-1)
                faces_area = faces_area.view(-1) * 100  # Scale face areas

                # Compute loss
                loss = weighted_ce_loss(S_out, y_adjusted, faces_area)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)  # Perform optimizer step
                scaler.update()  # Update the scaler for mixed precision
                optimizer.zero_grad()  # Reset gradients

            epoch_loss += loss.item()
            predictions = torch.argmax(S_out, dim=1)
            accuracy = calculate_accuracy(predictions, y_adjusted, faces_area)
            epoch_accuracy += accuracy

        # Store average loss and accuracy per epoch
        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(epoch_accuracy / len(train_loader))

        # Validation phase
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Save model
        torch.save(model.state_dict(), f'models/model_epoch_{epoch + 1}.pth')

        # Epoch summary
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)


# Evaluation Loop
def evaluate(model, loader, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    
    with torch.no_grad():
        for batch in loader:
            T_features, J_features, A_matrix, C_matrix, faces_area, y = [x.to(device) for x in batch[:-1]]
            
            # Forward pass
            S_out, _ = model(T_features, J_features, A_matrix, C_matrix)
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
            faces_area = faces_area * 100
            
            loss = weighted_ce_loss(S_out, y_adjusted, faces_area)
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(S_out, dim=1)
            targets = y_adjusted
            areas = faces_area
            accuracy = calculate_accuracy(predictions, targets, areas)
            total_accuracy += accuracy
    
    avg_loss = total_loss / len(loader)
    avg_accuracy = total_accuracy / len(loader)
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
            grad_norm = param.grad.norm()  # Calculate the norm of the gradients
            print(f"Layer: {name}, Grad Norm: {grad_norm.item()}")

            # You can set thresholds to detect vanishing or exploding gradients
            if grad_norm.item() < 1e-6:  # A small threshold for vanishing gradients
                print(f"Warning: Vanishing gradient detected in {name}")
            elif grad_norm.item() > 1e2:  # A large threshold for exploding gradients
                print(f"Warning: Exploding gradient detected in {name}")
      

def get_and_visualize_outputs(model, dataloader, device, save):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for i, data in enumerate(dataloader):
            T_features, J_features, A_matrix, C_matrix, faces_area, y = [x.to(device) for x in data[:-1]]

            mesh = preprocess_mesh(load_mesh(data[-1][i]), 1200)
            
            # Check for NaNs in input data
            if torch.isnan(T_features).any() or torch.isnan(J_features).any():
                print("NaNs found in input features!")

            # Forward pass
            S_out, P_out = model(T_features, J_features, A_matrix, C_matrix)
            S_out = S_out.view(-1, 4)  # Adjust as needed based on output shape
            S_out = torch.argmax(S_out, dim=1)

            # Loop through each mesh in the batch for visualization
            visualize_colored_mesh(mesh, S_out.cpu()[:len(mesh.triangles)].numpy(), f"{save}_{i}")
            
def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = MeshDataset(root='dataset', raw_mesh_folder='raw_meshes', seg_folder='seg')
    
    # Split into training and validation datasets
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
   
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Initialize model
    model = MeshTransformer(
        t_dim=28,
        J_dim=2412,
        P_dim=512,
        E_dim=1024,
        num_of_labels=4,
        num_of_encoder_layers=3,
        dropout=0.1,
        num_heads=8,
        add_norm=True,
        residual=True
    ).to(device)

    # Train the model
    num_epochs = 10
    learning_rate = 5e-5
    weight_decay = 0.01
    train(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay=weight_decay, device=device, accumulation_steps=6)
    
    # Evaluate the model on the validation set
    val_loss, val_accuracy = evaluate(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
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
    
    