import torch
import torch.nn as nn
import math

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.num_channels = num_channels
        self.reduction_ratio = reduction_ratio

        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels)
        )

    def forward(self, x):
        channel_avg = torch.mean(x, dim=2, keepdim=True)  
        channel_max, _ = torch.max(x, dim=2, keepdim=True) 
        combined = channel_avg + channel_max  
        combined = combined.squeeze(2)
        attention = self.mlp(combined)  # (batch_size, channels)
        attention = torch.sigmoid(attention).unsqueeze(2) # (batch_size, 1, 1, channels)
        attended_x = x * attention
        return attended_x, attention.squeeze()

class LearnableSTFT(nn.Module):

    def __init__(self, window_size, hop_size, dft_size=None):
        super(LearnableSTFT, self).__init__()


        self.window_size = window_size
        self.hop_size = hop_size
        self.dft_size = dft_size if dft_size is not None else window_size

        initial_window = 0.54 - 0.46 * torch.cos(
            2 * math.pi * torch.arange(window_size, dtype=torch.float32) / (window_size - 1)
        )

        self.window = nn.Parameter(initial_window)

        dft_matrix = self._create_dft_matrix(self.dft_size, self.window_size)
        
        self.register_buffer('dft_matrix', dft_matrix)

    def _create_dft_matrix(self, dft_size, window_size):

        k = torch.arange(dft_size).unsqueeze(1) # Shape: [dft_size, 1]
        n = torch.arange(window_size)            # Shape: [window_size]

        # Calculate the angle for the complex exponential
        angle = -2 * math.pi * k * n / dft_size

        # Use Euler's formula to create the complex matrix
        # e^(j*angle) = cos(angle) + j*sin(angle)
        dft_matrix = torch.complex(torch.cos(angle), torch.sin(angle))
        return dft_matrix

    def forward(self, signal):

        if signal.dim() == 1:
            signal = signal.unsqueeze(0).unsqueeze(0) # [1, 1, T]
        elif signal.dim() == 2:
            signal = signal.unsqueeze(1) # [B, 1, T]
        batch_size, num_channels, num_samples = signal.shape

        signal_reshaped = signal.reshape(batch_size * num_channels, num_samples)

        learnable_window = self.window
        frames = signal_reshaped.unfold(dimension=1, size=self.window_size, step=self.hop_size)

        num_frames_unfolded = frames.shape[1]
        expected_num_frames = int(math.ceil((num_samples - self.window_size) / self.hop_size)) + 1

        if num_frames_unfolded < expected_num_frames:
            padding_amount = (expected_num_frames - 1) * self.hop_size + self.window_size - num_samples
            padded_signal = torch.nn.functional.pad(signal_reshaped, (0, padding_amount))
            frames = padded_signal.unfold(1, self.window_size, self.hop_size)

        windowed_frames = frames * learnable_window

        BC, F, W = windowed_frames.shape # BC = batch_size * num_channels
        windowed_frames_reshaped = windowed_frames.reshape(BC * F, W)

        windowed_frames_complex = windowed_frames_reshaped.to(self.dft_matrix.dtype)
        stft_result_reshaped = self.dft_matrix @ windowed_frames_complex.T
        stft_result = stft_result_reshaped.T.reshape(batch_size, num_channels, F, self.dft_size)

        return stft_result

class Attention4D(nn.Module):

    def __init__(self, in_channels, time_frames, freq_bins, d_model=128, n_heads=4, d_ff=256, dropout=0.1):

        super().__init__()

        self.d_model = d_model
        self.time_frames = time_frames

        in_features = in_channels * freq_bins

        self.projection = nn.Linear(in_features, d_model)

        self.positional_encoding = nn.Parameter(torch.randn(1, time_frames, d_model))

        self.layernorm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.layernorm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):


        batch_size = x.shape[0]

        x = x.permute(0, 2, 1, 3)

        x = x.reshape(batch_size, self.time_frames, -1)

        x = self.projection(x) # Output: (batch, time_frames, d_model)

        x = x + self.positional_encoding

        x_norm1 = self.layernorm1(x)
        attn_output, _ = self.attention(x_norm1, x_norm1, x_norm1)
        x = x + self.dropout1(attn_output)

        x_norm2 = self.layernorm2(x)
        ff_output = self.feed_forward(x_norm2)
        x = x + self.dropout2(ff_output)

        return x
class Classifier(nn.Module):

    def __init__(self, in_features, num_classes):

        super().__init__()

        self.pooling = nn.AdaptiveAvgPool1d(1) # A flexible way to do global average pooling
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        
        x = x.permute(0, 2, 1)  # Shape -> (4, 128, 17)

        x = self.pooling(x).squeeze(2) # Shape -> (4, 128)

        output = self.classifier(x) # Shape -> (4, num_classes)

        return output
    
class EEGClassifier(nn.Module):

    def __init__(self, num_classes=4):
        super(EEGClassifier, self).__init__()
        self.channel_attn = ChannelAttention(num_channels = 22)
        self.learnable_stft = LearnableSTFT(window_size=250, hop_size=100)
        
        self.attention_module = Attention4D(
            in_channels=22,
            time_frames=14,
            freq_bins=250,
            d_model=128,
            n_heads=8
        )
        self.classifier = Classifier(in_features=128, num_classes=num_classes)

    def forward(self, x):

        x,_ = self.channel_attn(x)
        x = self.learnable_stft(x)
        x = torch.abs(x)
        x = self.attention_module(x)
        x = self.classifier(x)

        return x
    
    
if __name__ == '__main__':

    signal_length = 1751 # e.g., 1 second of audio at 16kHz
    batch_size = 4
    num_channels = 22

    dummy_input = torch.randn(batch_size, num_channels, signal_length)
    print(f"Initial input shape: {dummy_input.shape}\n")
    attention_module = EEGClassifier(num_classes=4)

    output = attention_module(dummy_input)

    print("--- Model Output ---")
    print(f"Final output shape: {output.shape}")
