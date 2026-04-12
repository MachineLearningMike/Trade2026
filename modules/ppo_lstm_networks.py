import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, nInFeatures, nOutFeatures, hidden_size=256, num_layers=2, dropout=0.5):
        """
        Args:
            nInFeatures: Input feature dimension (size of each element in sequence)
            nOutFeatures: Output feature dimension
            hidden_size: LSTM hidden size (default: 256)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout probability (default: 0.5)
        """
        super(BiLSTM, self).__init__()
        
        self.nInFeatures = nInFeatures
        self.nOutFeatures = nOutFeatures
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=nInFeatures,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq_len, features)
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer (2 * hidden_size because bidirectional)
        self.fc = nn.Linear(hidden_size * 2, nOutFeatures)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, nInFeatures)
        Returns:
            Output tensor of shape (batch, nOutFeatures)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # For bidirectional LSTM, hidden state shape: (2 * num_layers, batch, hidden_size)
        # We want the last layer's hidden states from both directions
        # hidden[-2] is forward hidden state of last layer
        # hidden[-1] is backward hidden state of last layer
        hidden_forward = hidden[-2, :, :]  # (batch, hidden_size)
        hidden_backward = hidden[-1, :, :]  # (batch, hidden_size)
        
        # Concatenate forward and backward hidden states
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)  # (batch, 2 * hidden_size)
        
        # Apply dropout
        hidden_concat = self.dropout(hidden_concat)
        
        # Final linear layer
        output = self.fc(hidden_concat)  # (batch, nOutFeatures)
        
        return output
    
class BiLSTMWithLastTimestep(nn.Module):
    def __init__(self, nInFeatures, nOutFeatures, hidden_size=256, num_layers=2, dropout=0.5):
        super(BiLSTMWithLastTimestep, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=nInFeatures,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # For using last timestep, we need to process both directions separately
        self.fc = nn.Linear(hidden_size * 2, nOutFeatures)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # lstm_out shape: (batch, seq_len, 2 * hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last timestep
        last_timestep = lstm_out[:, -1, :]  # (batch, 2 * hidden_size)
        
        # Apply dropout and linear layer
        output = self.fc(self.dropout(last_timestep))  # (batch, nOutFeatures)
        
        return output
    
class BiLSTMWithAttention(nn.Module):
    def __init__(self, nInFeatures, nOutFeatures, hidden_size=256, num_layers=2, dropout=0.5):
        super(BiLSTMWithAttention, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=nInFeatures,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, nOutFeatures)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # lstm_out shape: (batch, seq_len, 2 * hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        # Calculate attention weights
        attention_weights = torch.tanh(self.attention(lstm_out))  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch, seq_len, 1)
        
        # Apply attention weights
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, 2 * hidden_size)
        
        # Final output
        output = self.fc(self.dropout(attended_output))  # (batch, nOutFeatures)
        
        return output
    
    
if __name__ == "__main__":
    
    # Create model
    # model = BiLSTM(
    # model = BiLSTMWithLastTimestep(
    model = BiLSTMWithAttention(   
        nInFeatures=10,    # 10 input features per timestep
        nOutFeatures=5,    # 5 output features
        hidden_size=128,   # LSTM hidden size
        num_layers=2,      # 2 LSTM layers
        dropout=0.3        # 30% dropout
    )

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Example input
    batch_size = 32
    seq_len = 300
    nInFeatures = 10

    x = torch.randn(batch_size, seq_len, nInFeatures).to(device)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")        # (32, 50, 10)
    print(f"Output shape: {output.shape}")  # (32, 5)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")