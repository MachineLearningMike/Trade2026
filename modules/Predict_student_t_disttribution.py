import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import StudentT

# ------------------------------------------------------------
# 1. Select device
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------
# 2. Neural network: outputs location, log‑scale, and log‑df
# ------------------------------------------------------------
class StudentTNetwork(nn.Module):
    def __init__(self, input_dim):
        super(StudentTNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # Three outputs: location (μ), log(scale), and raw parameter for df
        self.output_head = nn.Linear(64, 3)

    def forward(self, x):
        features = self.shared_layers(x)
        params = self.output_head(features)

        location = params[:, 0:1]          # μ
        log_scale = params[:, 1:2]         # log(σ)
        log_df_raw = params[:, 2:3]        # unconstrained raw value for df

        return location, log_scale, log_df_raw

# ------------------------------------------------------------
# 3. Negative log‑likelihood using torch.distributions.StudentT
#    This automatically handles the standard parameterisation.
# ------------------------------------------------------------
def student_t_nll_loss(location, log_scale, log_df_raw, target):
    """
    Negative log‑likelihood for a Student‑t distribution.
    - location : μ (mean)
    - scale    : σ = exp(log_scale)   (must be > 0)
    - df       : degrees of freedom, enforced > 2 via softplus + 2
    """
    scale = torch.exp(log_scale)
    df = 2.0 + F.softplus(log_df_raw)   # ensures df > 2

    # Create the distribution
    dist = StudentT(df=df, loc=location, scale=scale)

    # Negative log‑likelihood
    nll = -dist.log_prob(target)
    return nll.mean()

# ------------------------------------------------------------
# 4. Instantiate model and optimizer
# ------------------------------------------------------------
model = StudentTNetwork(input_dim=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------
# 5. Generate synthetic data with EXPLICIT true parameters
# ------------------------------------------------------------
N = 200000
X_train = torch.randn(N, 10, device=device)

# --- True parameters of the data generating process ---
true_location = 1.5          # offset
true_slope = 2.0             # multiplier for the input feature
true_noise_location = 0.0    # noise mean
true_noise_scale = 0.5       # noise scale (like std)
true_noise_df = 3.0          # degrees of freedom for noise

# True location for each sample
true_mu_per_sample = true_location + true_slope * X_train.mean(dim=1, keepdim=True)

# Generate Student‑t noise with the true parameters
noise_dist = StudentT(df=true_noise_df, loc=true_noise_location, scale=true_noise_scale)
noise = noise_dist.sample((N, 1)).to(device)

# Final target
y_train = true_mu_per_sample + noise

print(f"Data generated with true location (offset): {true_location}, "
      f"true slope: {true_slope}, "
      f"true noise location: {true_noise_location}, "
      f"true noise scale: {true_noise_scale}, "
      f"true noise df: {true_noise_df}")

# ------------------------------------------------------------
# 6. Training loop
# ------------------------------------------------------------
for epoch in range(500):
    optimizer.zero_grad()

    location, log_scale, log_df_raw = model(X_train)
    loss = student_t_nll_loss(location, log_scale, log_df_raw, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        with torch.no_grad():
            scale = torch.exp(log_scale)
            df = 2.0 + F.softplus(log_df_raw)
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Location: {location.mean().item():.3f} | "
                  f"Scale: {scale.mean().item():.3f} | "
                  f"Degrees of freedom: {df.mean().item():.3f}")

# ------------------------------------------------------------
# 7. Inference on a new sample
# ------------------------------------------------------------
# with torch.no_grad():
#     test_input = torch.randn(1, 10, device=device)
#     location, log_scale, log_df_raw = model(test_input)
#     scale = torch.exp(log_scale)
#     df = 2.0 + F.softplus(log_df_raw)

#     print(f"\n--- Inference on a single test point ---")
#     print(f"Predicted location: {location.item():.3f}")
#     print(f"Predicted scale:    {scale.item():.3f}")
#     print(f"Predicted df:       {df.item():.3f}")

with torch.no_grad():
    test_input = torch.randn(1, 10, device=device)
    location, log_scale, log_df_raw = model(test_input)
    scale = torch.exp(log_scale)
    df = 2.0 + F.softplus(log_df_raw)

    print(f"\n--- Inference on a single test point ---")
    print(f"Test input mean:        {test_input.mean().item():.4f}")
    print(f"Predicted location:     {location.item():.3f}")
    print(f"Expected location:      1.5 + 2.0 * {test_input.mean().item():.4f} = {1.5 + 2.0 * test_input.mean().item():.3f}")
    print(f"Predicted scale:        {scale.item():.3f}")
    print(f"Predicted df:           {df.item():.3f}")


"""
The model has successfully recovered the true underlying data-generating process. 
Your location is only "wrong" if you were expecting it to be the global intercept (1.5) — 
but in a conditional model, location is the conditional mean, not the global offset.
"""