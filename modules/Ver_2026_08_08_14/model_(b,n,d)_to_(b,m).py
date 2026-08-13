import torch
import torch.nn as nn

# Option 1: Global Pooling + Projection (Most Common)

# Input: (batch_size, N, D)
x = torch.randn(32, 100, 64)  # batch=32, N=100, D=64

# Global pooling over N dimension
pooled = x.mean(dim=1)  # (batch_size, D) -> (32, 64)

# Project to 3 dimensions
projection = nn.Linear(64, 3)
output = projection(pooled)  # (batch_size, 3)


# Option 2: Multi-Head Aggregation (Preserve More Info)

class MultiHeadAggregator(nn.Module):
    def __init__(self, D, num_heads=3, output_dim=3):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D, D//2),
                nn.ReLU(),
                nn.Linear(D//2, 1)
            ) for _ in range(num_heads)
        ])
        
    def forward(self, x):
        # x: (batch_size, N, D)
        outputs = []
        for head in self.heads:
            # Each head produces (batch_size, N, 1)
            head_out = head(x)  # (batch_size, N, 1)
            # Aggregate over N
            head_out = head_out.mean(dim=1)  # (batch_size, 1)
            outputs.append(head_out)
        
        # Concatenate heads
        return torch.cat(outputs, dim=1)  # (batch_size, 3)

# Usage
aggregator = MultiHeadAggregator(D=64, num_heads=3)
output = aggregator(x)  # (batch_size, 3)


# Option 3: Attention-Based Aggregation

class AttentionAggregator(nn.Module):
    def __init__(self, D, output_dim=3):
        super().__init__()
        # Learnable query for attention
        self.query = nn.Parameter(torch.randn(1, 1, D))  # (1, 1, D)
        self.projection = nn.Linear(D, output_dim)
        
    def forward(self, x):
        # x: (batch_size, N, D)
        
        # Compute attention scores
        # query: (1, 1, D) -> expanded to (batch_size, 1, D)
        query = self.query.expand(x.size(0), -1, -1)  # (batch_size, 1, D)
        
        # Dot product attention
        scores = torch.matmul(query, x.transpose(1, 2))  # (batch_size, 1, N)
        scores = scores / (x.size(-1) ** 0.5)  # Scale
        weights = torch.softmax(scores, dim=-1)  # (batch_size, 1, N)
        
        # Weighted sum
        aggregated = torch.matmul(weights, x)  # (batch_size, 1, D)
        aggregated = aggregated.squeeze(1)  # (batch_size, D)
        
        # Project to 3 dimensions
        output = self.projection(aggregated)  # (batch_size, 3)
        return output

# Usage
aggregator = AttentionAggregator(D=64, output_dim=3)
output = aggregator(x)  # (batch_size, 3)


# Option 4: Multiple Pooling Operations + Concatenation

class MultiPoolAggregator(nn.Module):
    def __init__(self, D, output_dim=3):
        super().__init__()
        # Different pooling operations
        self.pooling_ops = {
            'mean': lambda x: x.mean(dim=1),
            'max': lambda x: x.max(dim=1)[0],
            'std': lambda x: x.std(dim=1),
            'median': lambda x: x.median(dim=1)[0],
            'sum': lambda x: x.sum(dim=1)
        }
        # Project concatenated features to 3
        self.projection = nn.Linear(D * 3, output_dim)
        
    def forward(self, x):
        # Use 3 different pooling operations
        pooled = [
            self.pooling_ops['mean'](x),  # (batch_size, D)
            self.pooling_ops['max'](x),   # (batch_size, D)
            self.pooling_ops['std'](x)    # (batch_size, D)
        ]
        # Concatenate
        concatenated = torch.cat(pooled, dim=1)  # (batch_size, 3*D)
        # Project to 3
        return self.projection(concatenated)  # (batch_size, 3)

# Usage
aggregator = MultiPoolAggregator(D=64)
output = aggregator(x)  # (batch_size, 3)


# Option 5: Learnable Transformation (Keep N dimension in latent space)

class LatentAggregator(nn.Module):
    def __init__(self, N, D, latent_dim=128, output_dim=3):
        super().__init__()
        # First, transform N*D to latent
        self.flatten = nn.Flatten()  # (batch_size, N*D)
        self.encoder = nn.Sequential(
            nn.Linear(N * D, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim)
        )
        
    def forward(self, x):
        # x: (batch_size, N, D)
        x = x.reshape(x.size(0), -1)  # (batch_size, N*D)
        return self.encoder(x)  # (batch_size, 3)

# Usage
aggregator = LatentAggregator(N=100, D=64)
output = aggregator(x)  # (batch_size, 3)


# Option 6: Self-Attention + Classification (Most Expressive)
# Can the attention compensate for the brutal lossy mean-ing?

class TransformerAggregator(nn.Module):
    def __init__(self, D, output_dim=3, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(D, num_heads, batch_first=True)
        self.projection = nn.Linear(D, output_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: (batch_size, N, D)
        # Self-attention
        attn_out, _ = self.attention(x, x, x)  # (batch_size, N, D)
        
        # Aggregate over N dimension
        pooled = attn_out.mean(dim=1)  # (batch_size, D)
        # or: pooled = attn_out.max(dim=1)[0]  # (batch_size, D)
        
        # Project to 3
        return self.projection(pooled)  # (batch_size, 3)


# Option 7: Learn Multiple "Queries" for Different Aspects

class MultiQueryAggregator(nn.Module):
    def __init__(self, D, num_queries=3, output_dim=3):
        super().__init__()
        # Learn different queries to capture different aspects
        self.queries = nn.Parameter(torch.randn(num_queries, 1, D))
        self.projection = nn.Linear(D, output_dim)
        
    def forward(self, x):
        # x: (batch_size, N, D)
        # queries: (3, 1, D) -> expand to (3, batch_size, 1, D)
        queries = self.queries.unsqueeze(1)  # (3, 1, D) -> (3, 1, 1, D)
        queries = queries.expand(-1, x.size(0), 1, -1)  # (3, batch_size, 1, D)
        
        outputs = []
        for i in range(queries.size(0)):
            q = queries[i]  # (batch_size, 1, D)
            scores = torch.matmul(q, x.transpose(1, 2))  # (batch_size, 1, N)
            weights = torch.softmax(scores / (D ** 0.5), dim=-1)
            aggregated = torch.matmul(weights, x).squeeze(1)  # (batch_size, D)
            outputs.append(aggregated)
        
        # Each query produces (batch_size, D) -> stack to (batch_size, 3, D)
        stacked = torch.stack(outputs, dim=1)  # (batch_size, 3, D)
        # Project each to 1 dimension
        projected = self.projection(stacked)  # (batch_size, 3, 1)
        return projected.squeeze(-1)  # (batch_size, 3)

"""
Recommendation:
For simplicity: Use Option 1 (Global Pooling + Linear)

For better performance: Use Option 3 (Attention Aggregator)

For most expressiveness: Use Option 6 (Transformer Aggregator)

For interpretability: Use Option 7 (Multi-Query Aggregator)

The choice depends on:

N: How many elements you're aggregating over

D: Feature dimension

Compute budget: More complex = more compute

Available data: Complex models need more data
"""