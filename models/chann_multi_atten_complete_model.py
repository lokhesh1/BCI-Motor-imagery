import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid() # Sigmoid activation to scale weights between 0 and 1

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))

        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(-1).unsqueeze(-1)

class EEGDecoder(nn.Module):
    def __init__(self, num_channels=22, freq_bins=64, time_bins=76, num_classes=4,
                 mha_num_heads=4, mha_dropout=0.1, channel_attention_reduction=4):
        super(EEGDecoder, self).__init__()

        self.num_channels = num_channels
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        self.num_classes = num_classes

        self.channel_attention = ChannelAttention(num_channels, reduction_ratio=channel_attention_reduction)

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=time_bins,      # The embedding dimension for MHA, must match the last dimension of input
            num_heads=mha_num_heads,  # Number of parallel attention heads
            dropout=mha_dropout,      # Dropout rate for attention weights
            batch_first=False         # Specifies input format: (seq_len, batch_size, embed_dim)
        )

        flattened_size = num_channels * freq_bins * time_bins
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # Flattens the input tensor (N, C, F, T) into (N, C*F*T)
            nn.Linear(flattened_size, 512), # First fully connected layer with 512 hidden units
            nn.ReLU(),                      # ReLU activation function
            nn.Dropout(0.5),                # Dropout for regularization to prevent overfitting
            nn.Linear(512, num_classes)     # Output layer, mapping to the number of classification classes
        )

    def forward(self, x):
        N, C, F, T = x.shape

        channel_attn_weights = self.channel_attention(x) # Shape: (N, C, 1, 1)
        x = x * channel_attn_weights                     # Resulting shape: (N, C, F, T)

        x_reshaped_for_mha = x.view(N * C, F, T).permute(1, 0, 2)

        mha_output, _ = self.multihead_attention(
            query=x_reshaped_for_mha,
            key=x_reshaped_for_mha,
            value=x_reshaped_for_mha
        ) # Output shape: (F, N*C, T)

        x = mha_output.permute(1, 0, 2).view(N, C, F, T)

        logits = self.classifier(x) # Output shape: (N, num_classes)

        return logits

if __name__ == "__main__":
    num_channels = 22
    freq_bins = 64
    time_bins = 76
    num_classes = 4
    batch_size = 16 # Example batch size

    dummy_input = torch.randn(batch_size, num_channels, freq_bins, time_bins)
    print(f"Dummy input shape: {dummy_input.shape}")

    model = EEGDecoder(
        num_channels=num_channels,
        freq_bins=freq_bins,
        time_bins=time_bins,
        num_classes=num_classes
    )
    print("\nModel Architecture:")
    print(model)

    output_logits = model(dummy_input)
    print(f"\nOutput logits shape: {output_logits.shape}")

    expected_output_shape = (batch_size, num_classes)
    assert output_logits.shape == expected_output_shape, \
        f"Output shape mismatch! Expected {expected_output_shape}, got {output_logits.shape}"
    print("Output shape is correct!")

    probabilities = F.softmax(output_logits, dim=1)
    print(f"Example probabilities for the first sample: {probabilities[0]}")

