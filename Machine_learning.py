import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt  # Add this import



# File paths
species_info_file = r""
distance_matrix_file = r""
output_graph_file = r""

# Hardcoded Mappings
media_mapping = {
    '275': 0, 'BBE': 1, 'BCYE': 2, 'BHI': 3, 'BKV': 4, 'BR': 5, 'CHEMO': 6, 'CNA': 7, 'COL': 8, 'COLB': 9,
    'DCM': 10, 'FAA': 11, 'FCC': 12, 'FCCR': 13, 'FCR': 14, 'FLG': 15, 'FMU': 16, 'FNB': 17, 'FNBU': 18,
    'GAM': 19, 'GC': 20, 'GMM': 21, 'JVN': 22, 'LGXC': 23, 'M17': 24, 'MB': 25, 'MH': 26, 'MPYG': 27,
    'MRS': 28, 'MSAB': 29, 'MTM': 30, 'Medium 1': 31, 'Medium 104': 32, 'Medium 104b': 33, 'Medium 104c': 34,
    'Medium 11': 35, 'Medium 110': 36, 'Medium 110a': 37, 'Medium 1203': 38, 'Medium 1203a': 39,
    'Medium 1611': 40, 'Medium 1668': 41, 'Medium 1713': 42, 'Medium 1715': 43, 'Medium 215': 44,
    'Medium 215c': 45, 'Medium 220': 46, 'Medium 328': 47, 'Medium 339': 48, 'Medium 339a': 49,
    'Medium 381': 50, 'Medium 414': 51, 'Medium 429': 52, 'Medium 514': 53, 'Medium 535': 54,
    'Medium 535a': 55, 'Medium 58': 56, 'Medium 641': 57, 'Medium 693': 58, 'Medium 78': 59,
    'Medium 840': 60, 'Medium 92': 61, 'PDA': 62, 'PTM': 63, 'R2A': 64, 'RCM': 65, 'RMD': 66, 'SAB': 67,
    'SDA': 68, 'SKV': 69, 'STSA': 70, 'TM': 71, 'TSAB': 72, 'WC': 73, 'YCFA': 74, 'YPD': 75, 'YPDC': 76, None: 77
}
condition_mapping = {
    '45': 0, '651H': 1, '6520M': 2, '851H': 3, '8520M': 4, 'BG11': 5, 'BLG': 6, 'BRF': 7, 'ETOH': 8,
    'NoCondition': 9, 'OTEB': 10, None: 11
}
atmosphere_mapping = {
    'AER': 1, 'AN': 2, None: 0
}

reverse_media_mapping = {v: k for k, v in media_mapping.items()}
reverse_condition_mapping = {v: k for k, v in condition_mapping.items()}
reverse_atmosphere_mapping = {v: k for k, v in atmosphere_mapping.items()}

# 1. Load species data
df_species = pd.read_csv(species_info_file)

# Apply hardcoded mappings
df_species['Media'] = df_species['Media'].map(media_mapping).fillna(77).astype(int)
df_species['Condition'] = df_species['Condition'].map(condition_mapping).fillna(11).astype(int)
df_species['Atmosphere'] = df_species['Atmosphere'].map(atmosphere_mapping).fillna(0).astype(int)

# Normalize numerical columns
if 'Temperature' in df_species.columns:
    df_species['Temperature'] = (
        df_species['Temperature']
        .replace(r'[^0-9.]', '', regex=True)
        .astype(float)
        .fillna(37)
    )
    scaler = StandardScaler()
    df_species[['Temperature']] = scaler.fit_transform(df_species[['Temperature']])

# Split data into train and test sets
train_species, test_species = train_test_split(df_species, test_size=0.2, stratify=df_species['Media'], random_state=42)

# 2. Load distance matrix
distance_matrix = pd.read_csv(distance_matrix_file, index_col=0)
print(f"Distance matrix loaded: {distance_matrix.shape}")

# Filter distance matrix for training and testing
train_distance_matrix = distance_matrix.loc[train_species['rank_species'], train_species['rank_species']]
test_distance_matrix = distance_matrix.loc[test_species['rank_species'], test_species['rank_species']]
print(f"Filtered training distance matrix: {train_distance_matrix.shape}")
print(f"Filtered testing distance matrix: {test_distance_matrix.shape}")

# 3. Create edge_index and edge_attr from the distance matrix
def create_edges_from_distance_matrix(distance_matrix, threshold=0.5):
    """
    Converts a distance matrix into edge indices and edge attributes.
    """
    edge_index = []
    edge_attr = []

    for i, source in enumerate(distance_matrix.index):
        for j, target in enumerate(distance_matrix.columns):
            if source != target and distance_matrix.iloc[i, j] < threshold:
                edge_index.append((i, j))
                edge_attr.append(distance_matrix.iloc[i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr

train_edge_index, train_edge_attr = create_edges_from_distance_matrix(train_distance_matrix)
test_edge_index, test_edge_attr = create_edges_from_distance_matrix(test_distance_matrix)

# 4. Create node features
train_node_features = torch.tensor(train_species.iloc[:, 1:].values, dtype=torch.float)
test_node_features = torch.tensor(test_species.iloc[:, 1:].values, dtype=torch.float)

# 5. Create PyTorch Geometric Data objects
train_data = Data(x=train_node_features, edge_index=train_edge_index, edge_attr=train_edge_attr)
test_data = Data(x=test_node_features, edge_index=test_edge_index, edge_attr=test_edge_attr)

# Save graph data for reuse
torch.save(train_data, output_graph_file.replace(".pt", "_train.pt"))
torch.save(test_data, output_graph_file.replace(".pt", "_test.pt"))
print(f"Graph data saved to: {output_graph_file}")

# 6. Define GNN-VAE model
class GNNVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(GNNVAE, self).__init__()
        self.encoder_gnn = GCNConv(input_dim, hidden_dim)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gnn = GCNConv(hidden_dim, input_dim)
        self.condition_decoder = nn.Linear(latent_dim, output_dim)  # For predicting conditions

    def encode(self, x, edge_index, edge_attr):
        x = self.encoder_gnn(x, edge_index).relu()
        z = self.encoder_fc(x)
        return z

    def decode(self, z, edge_index, edge_attr):
        h = self.decoder_fc(z).relu()
        x_recon = self.decoder_gnn(h, edge_index)
        return x_recon

    def predict_conditions(self, z):
        return self.condition_decoder(z)

    def forward(self, x, edge_index, edge_attr):
        z = self.encode(x, edge_index, edge_attr)
        x_recon = self.decode(z, edge_index, edge_attr)
        predicted_conditions = self.predict_conditions(z)
        return x_recon, z, predicted_conditions

# 7. Initialize Model
input_dim = train_data.num_node_features
hidden_dim = 64
latent_dim = 32
output_dim = 3  # Media, Condition, Atmosphere
model = GNNVAE(input_dim, hidden_dim, latent_dim, output_dim)

# 8. Train the model
def train_model(model, data, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn_reconstruction = nn.MSELoss()
    loss_fn_conditions = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        x_recon, z, predicted_conditions = model(data.x, data.edge_index, data.edge_attr)

        # Extract true conditions
        true_conditions = data.x[:, -output_dim:]  # Assuming last columns are the target features
        true_conditions = true_conditions.argmax(dim=1).long()

        # Compute losses
        loss_reconstruction = loss_fn_reconstruction(x_recon, data.x)
        loss_conditions = loss_fn_conditions(predicted_conditions, true_conditions)
        loss = loss_reconstruction + loss_conditions

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Reconstruction Loss: {loss_reconstruction.item()}, Condition Loss: {loss_conditions.item()}")

train_model(model, train_data)

# 9. Evaluate the model
def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        x_recon, z, predicted_conditions = model(data.x, data.edge_index, data.edge_attr)
        true_conditions = data.x[:, -output_dim:].argmax(dim=1).long()
        predicted_labels = predicted_conditions.argmax(dim=1)
        accuracy = (predicted_labels == true_conditions).sum().item() / true_conditions.size(0)
        mse_loss = nn.MSELoss()(x_recon, data.x).item()
    print(f"Evaluation - Reconstruction Loss: {mse_loss:.4f}, Condition Prediction Accuracy: {accuracy:.4f}")
    return mse_loss, accuracy

evaluate_model(model, test_data)

# 10. Sample new conditions
def sample_new_conditions(model, num_samples=10):
    model.eval()
    with torch.no_grad():
        # Sample from a normal distribution with variance
        z_samples = torch.randn(num_samples, model.encoder_fc.out_features) * 1.5  # Adjust variance
        new_conditions = model.predict_conditions(z_samples)
        return new_conditions

new_conditions = sample_new_conditions(model, num_samples=5)
print("New optimized conditions:", new_conditions)

# 11. Decode new conditions
def decode_conditions(new_conditions):
    decoded_conditions = {"Media": [], "Condition": [], "Atmosphere": []}
    for row in new_conditions.numpy():
        media_idx = int(round(row[0]))
        condition_idx = int(round(row[1]))
        atmosphere_idx = int(round(row[2]))

        media_idx = max(0, min(media_idx, len(reverse_media_mapping) - 1))
        condition_idx = max(0, min(condition_idx, len(reverse_condition_mapping) - 1))
        atmosphere_idx = max(0, min(atmosphere_idx, len(reverse_atmosphere_mapping) - 1))

        decoded_conditions["Media"].append(reverse_media_mapping.get(media_idx, "Unknown"))
        decoded_conditions["Condition"].append(reverse_condition_mapping.get(condition_idx, "Unknown"))
        decoded_conditions["Atmosphere"].append(reverse_atmosphere_mapping.get(atmosphere_idx, "Unknown"))

    return pd.DataFrame(decoded_conditions)

# Decode the new conditions
df_predictions = decode_conditions(new_conditions)

# Print the decoded predictions
print("Decoded Predictions:")
print(df_predictions)

print("Unique values in Media:", df_species['Media'].unique())
print("Unique values in Condition:", df_species['Condition'].unique())
print("Unique values in Atmosphere:", df_species['Atmosphere'].unique())



# Specify the path where you want to save the decoded predictions
decoded_predictions_file = r"decoded_predictions.csv"

# Save the decoded predictions DataFrame as a CSV file
df_predictions.to_csv(decoded_predictions_file, index=False)

print(f"Decoded predictions have been saved to: {decoded_predictions_file}")


# Ensure the evaluation directory exists
evaluation_dir = r""
os.makedirs(evaluation_dir, exist_ok=True)

# File paths for saving evaluation results
results_file = os.path.join(evaluation_dir, "evaluation_results.txt")
decoded_file = os.path.join(evaluation_dir, "decoded_predictions.csv")
save_fig_path = os.path.join(evaluation_dir, "evaluation_metrics.png")

# 9. Evaluate the model and save results
def evaluate_model(model, data, results_file, decoded_file, save_fig_path):
    model.eval()
    with torch.no_grad():
        x_recon, z, predicted_conditions = model(data.x, data.edge_index, data.edge_attr)
        true_conditions = data.x[:, -output_dim:].argmax(dim=1).long()
        predicted_labels = predicted_conditions.argmax(dim=1)
        
        # Compute Metrics
        accuracy = (predicted_labels == true_conditions).sum().item() / true_conditions.size(0)
        mse_loss = nn.MSELoss()(x_recon, data.x).item()
        
        print(f"Evaluation - Reconstruction Loss: {mse_loss:.4f}, Condition Prediction Accuracy: {accuracy:.4f}")
        
        # Save metrics
        with open(results_file, "w") as f:
            f.write(f"Reconstruction Loss: {mse_loss:.4f}\n")
            f.write(f"Condition Prediction Accuracy: {accuracy:.4f}\n")
        
        # Decode predictions
        decoded_conditions = decode_conditions(predicted_conditions.cpu())
        decoded_conditions.to_csv(decoded_file, index=False)
        
        # Visualization with adjusted annotations
        plt.figure(figsize=(8, 6))
        plt.bar(["Reconstruction Loss", "Accuracy"], [mse_loss, accuracy], color=['blue', 'green'])
        plt.ylabel("Metric Values", fontsize=12)
        plt.xlabel("Evaluation Metrics", fontsize=12)
        plt.title("Model Evaluation: Reconstruction Loss vs Condition Prediction Accuracy", fontsize=14)
        plt.ylim(0, max(mse_loss, 1.0) + 5)  # Adjust y-axis for better spacing

        # Annotate Reconstruction Loss (text in the middle of the bar)
        plt.text(0, mse_loss / 2, f"{mse_loss:.4f}", ha='center', va='center', fontsize=10, color='white')

        # Annotate Accuracy (text on top of the bar)
        plt.text(1, accuracy + 0.5, f"{accuracy:.4f}", ha='center', va='bottom', fontsize=10, color='green')

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(save_fig_path)
        plt.close()


# Call the evaluation function
evaluate_model(
    model,
    test_data,
    results_file,
    decoded_file,
    save_fig_path,
)
