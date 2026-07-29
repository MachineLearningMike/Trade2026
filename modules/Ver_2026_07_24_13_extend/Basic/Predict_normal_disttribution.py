import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------------------------------------------
# 1. Select device: GPU if available, else CPU
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------
# 2. Model definition (unchanged)
# ------------------------------------------------------------
class MeanVarianceNetwork(nn.Module):
    def __init__(self, input_dim):
        super(MeanVarianceNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(64, 1)
        self.log_var_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared_layers(x)
        mu = self.mean_head(features)
        log_var = self.log_var_head(features)
        return mu, log_var

# ------------------------------------------------------------
# 3. Loss function (unchanged): negative- log-likelihood for Gaussian
# ------------------------------------------------------------
def gaussian_nll_loss(mu, log_var, target):
    var = torch.exp(log_var)
    loss = 0.5 * (log_var + ((target - mu) ** 2) / var)
    return loss.mean()

# ------------------------------------------------------------
# 4. Instantiate model and move it to the device
# ------------------------------------------------------------
model = MeanVarianceNetwork(input_dim=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------
# 5. Generate training data directly on the device
# ------------------------------------------------------------
N = 200000
X_train = torch.randn(N, 10, device=device)

mean, std = 1.5, 0.5
y_train = mean + 2.0 * X_train.mean(dim=1).unsqueeze(1) + std * torch.randn(N, 1, device=device)

# ------------------------------------------------------------
# 6. Training loop (data and model are already on the device)
# ------------------------------------------------------------
for epoch in range(500):
    optimizer.zero_grad()
    
    mu, log_var = model(X_train)
    loss = gaussian_nll_loss(mu, log_var, y_train)
    
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        predicted_std = torch.sqrt(torch.exp(log_var))
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
              f"Predicted Mean: {mu.mean().item():.3f}, "
              f"Predicted std: {predicted_std.mean().item():.3f}")

# ------------------------------------------------------------
# 7. Inference on a new sample (also on the correct device)
# ------------------------------------------------------------
with torch.no_grad():
    test_input = torch.randn(1, 10, device=device)
    mu, log_var = model(test_input)
    predicted_std = torch.sqrt(torch.exp(log_var))
    print(f"\nPredicted Mean: {mu.item():.3f}, Predicted std: {predicted_std.item():.3f}")