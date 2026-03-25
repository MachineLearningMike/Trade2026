import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def bce_loss(y_true, y_pred):
    """Binary Cross Entropy loss"""
    # Avoid numerical issues
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Create meshgrid for prediction values (x-axis) and true labels (y-axis)
# Actually, we'll create: x = prediction, y = true label (0 or 1)
# But to make it 3D, we'll use a gradient along y-axis

# Method 1: Surface plot showing loss for prediction [0,1] and true label [0,1]
# But true label is binary, so we'll create two surfaces side by side

pred = np.linspace(0.001, 0.999, 200)

# Calculate loss for both true labels
loss_y0 = bce_loss(0, pred)  # When true label = 0
loss_y1 = bce_loss(1, pred)  # When true label = 1

# Create 3D plot with prediction on x-axis, true label category on y-axis
# We'll use a surface plot with y-axis representing true label (0 and 1)

fig = go.Figure()

# Add surface for true label = 0
pred_grid, y0_grid = np.meshgrid(pred, [0])
loss_grid_y0 = bce_loss(0, pred_grid)

fig.add_trace(go.Surface(
    x=pred_grid,
    y=y0_grid,
    z=loss_grid_y0,
    name='True Label = 0',
    colorscale='Viridis',
    showscale=True,
    opacity=0.9,
    colorbar=dict(title="Loss Value"),
    contours=dict(
        z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
    )
))

# Add surface for true label = 1
pred_grid, y1_grid = np.meshgrid(pred, [1])
loss_grid_y1 = bce_loss(1, pred_grid)

fig.add_trace(go.Surface(
    x=pred_grid,
    y=y1_grid,
    z=loss_grid_y1,
    name='True Label = 1',
    colorscale='Viridis',
    showscale=False,
    opacity=0.9
))

# Add lines to highlight minimum points
min_y0_idx = np.argmin(loss_y0)
min_y1_idx = np.argmin(loss_y1)

# Add scatter points at minima
fig.add_trace(go.Scatter3d(
    x=[pred[min_y0_idx], pred[min_y1_idx]],
    y=[0, 1],
    z=[loss_y0[min_y0_idx], loss_y1[min_y1_idx]],
    mode='markers',
    marker=dict(size=8, color='red', symbol='circle'),
    name='Minima (Loss = 0)'
))

# Add annotation lines from minima
for y_val, pred_val, loss_val in [(0, pred[min_y0_idx], loss_y0[min_y0_idx]),
                                   (1, pred[min_y1_idx], loss_y1[min_y1_idx])]:
    fig.add_trace(go.Scatter3d(
        x=[pred_val, pred_val],
        y=[y_val, y_val],
        z=[loss_val, 5],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=False
    ))

# Update layout for better visualization
fig.update_layout(
    title=dict(
        text="<b>Binary Cross Entropy (BCE) Loss Function</b><br><sup>Interactive 3D Visualization</sup>",
        font=dict(size=20),
        x=0.5
    ),
    scene=dict(
        xaxis=dict(
            title="<b>Predicted Probability (p)</b>",
            range=[0, 1],
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=['0', '0.25', '0.5', '0.75', '1'],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="<b>True Label (y)</b>",
            tickvals=[0, 1],
            ticktext=['Class 0', 'Class 1'],
            range=[-0.2, 1.2],
            gridcolor='lightgray'
        ),
        zaxis=dict(
            title="<b>Loss Value</b>",
            range=[0, 5],
            gridcolor='lightgray'
        ),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        ),
        aspectmode='manual',
        aspectratio=dict(x=1.5, y=0.8, z=1)
    ),
    width=1000,
    height=800,
    hovermode='closest'
)

# Add annotations for key points
fig.add_annotation(
    x=0.5,
    y=0.5,
    text="<b>Key Insight:</b> Loss is minimized when prediction matches true label",
    showarrow=False,
    font=dict(size=12),
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="black",
    borderwidth=1
)

fig.show()

# Create a more interactive version with slider for true label
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("<b>BCE Loss Surface (3D)</b>", "<b>Cross-Section at Fixed Predictions</b>"),
    specs=[[{'type': 'scene'}, {'type': 'xy'}]]
)

# Add 3D surface to first subplot
pred_grid_full = np.linspace(0.001, 0.999, 100)
y_grid_full = np.linspace(0, 1, 100)
P, Y = np.meshgrid(pred_grid_full, y_grid_full)
Z = bce_loss(Y, P)

fig2.add_trace(
    go.Surface(x=P, y=Y, z=Z, colorscale='Viridis', colorbar=dict(title="Loss")),
    row=1, col=1
)

# Add 2D cross-sections for different prediction values
pred_values = [0.1, 0.3, 0.5, 0.7, 0.9]
y_range = np.linspace(0, 1, 200)

for pred_val in pred_values:
    loss_curve = bce_loss(y_range, pred_val)
    fig2.add_trace(
        go.Scatter(x=y_range, y=loss_curve, mode='lines', name=f'p = {pred_val}',
                  line=dict(width=2)),
        row=1, col=2
    )

# Add perfect prediction line (loss = 0 at y = p)
fig2.add_trace(
    go.Scatter(x=[0, 1], y=[0, 0], mode='lines', name='Minimum (Loss=0)',
              line=dict(color='red', width=2, dash='dash')),
    row=1, col=2
)

fig2.update_layout(
    title_text="<b>Binary Cross Entropy: Multiple Views</b>",
    height=600,
    width=1200,
    showlegend=True
)

fig2.update_xaxes(title_text="<b>True Label (y)</b>", row=1, col=2)
fig2.update_yaxes(title_text="<b>Loss Value</b>", row=1, col=2)

fig2.show()

# Create a 3D scatter plot with color mapping for different (y, p) pairs
# This creates a more "explorable" version with points you can rotate and hover

# Generate sample points
np.random.seed(42)
n_points = 500
y_true = np.random.choice([0, 1], n_points)
y_pred = np.random.beta(2, 2, n_points)  # Beta distribution centered around 0.5
loss = bce_loss(y_true, y_pred)

fig3 = go.Figure(data=[go.Scatter3d(
    x=y_pred,
    y=y_true,
    z=loss,
    mode='markers',
    marker=dict(
        size=5,
        color=loss,
        colorscale='Viridis',
        colorbar=dict(title="Loss Value"),
        showscale=True,
        opacity=0.7
    ),
    text=[f"Prediction: {p:.3f}<br>True Label: {y}<br>Loss: {l:.3f}" 
          for p, y, l in zip(y_pred, y_true, loss)],
    hoverinfo='text'
)])

fig3.update_layout(
    title=dict(
        text="<b>BCE Loss: Random Sample Points</b><br><sup>Hover over points to see values</sup>",
        font=dict(size=18),
        x=0.5
    ),
    scene=dict(
        xaxis_title="<b>Predicted Probability (p)</b>",
        yaxis_title="<b>True Label (y)</b>",
        zaxis_title="<b>Loss Value</b>",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(tickvals=[0, 1], ticktext=['0', '1']),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    width=900,
    height=700
)

fig3.show()

# Print mathematical insights
print("\n" + "="*60)
print("📊 BINARY CROSS ENTROPY - KEY INSIGHTS")
print("="*60)
print("\n📐 Formula:")
print("   BCE = -[y·log(p) + (1-y)·log(1-p)]")
print("\n📍 Key Points:")
print("   • Minimum loss (0): When p = y")
print("   • Maximum loss (∞): When prediction is opposite of true label")
print("   • Loss at p=0.5 when y=1: -log(0.5) = 0.693")
print("\n🎯 Properties:")
print("   • Convex function → unique global minimum")
print("   • Gradient: ∂BCE/∂p = (p - y) / [p(1-p)]")
print("   • Gradient w.r.t logits: p - y (very simple!)")
print("\n💡 Interactive Features:")
print("   • Rotate 3D plots by clicking and dragging")
print("   • Zoom with scroll wheel")
print("   • Hover to see exact values")
print("   • Double-click legend to isolate traces")
print("="*60)