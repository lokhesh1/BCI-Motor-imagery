import torch
import torch.nn as nn
import math

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
        
        angle = -2 * math.pi * k * n / dft_size
        
        dft_matrix = torch.complex(torch.cos(angle), torch.sin(angle))
        
        return dft_matrix

    def forward(self, signal):
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        
        learnable_window = self.window

        frames = signal.unfold(dimension=1, size=self.window_size, step=self.hop_size)
        
        num_samples = signal.shape[-1]
        num_frames_unfolded = frames.shape[1]
        expected_num_frames = int(math.ceil((num_samples - self.window_size) / self.hop_size)) + 1
        
        if num_frames_unfolded < expected_num_frames:
            padding_amount = (expected_num_frames - 1) * self.hop_size + self.window_size - num_samples
            padded_signal = torch.nn.functional.pad(signal, (0, padding_amount))
            frames = padded_signal.unfold(1, self.window_size, self.hop_size)

        windowed_frames = frames * learnable_window

        B, F, W = windowed_frames.shape
        windowed_frames_reshaped = windowed_frames.reshape(B * F, W)
        
        stft_result = self.dft_matrix @ windowed_frames_reshaped.T
        
        stft_result = stft_result.T.reshape(B, F, self.dft_size)

        return stft_result

if __name__ == '__main__':
    window_size = 400
    hop_size = 160
    signal_length = 16000 # e.g., 1 second of audio at 16kHz
    
    input_signal = torch.randn(signal_length)
    
    stft_layer = LearnableSTFT(window_size, hop_size)
    
    print("Initial window (first 5 values):")
    print(stft_layer.window.data[:5])
    
    # --- Perform a forward pass ---
    stft_output = stft_layer(input_signal)
    
    print("\nShape of the STFT output tensor:")
    print(stft_output.shape) 
    print(f"Output is a complex tensor: {stft_output.is_complex()}")

    optimizer = torch.optim.SGD(stft_layer.parameters(), lr=0.01)
    dummy_target = torch.randn_like(stft_output, dtype=torch.complex64)
    
    loss = torch.mean((stft_output.abs() - dummy_target.abs())**2)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("\nWindow after one optimization step (first 5 values):")
    print(stft_layer.window.data[:5])
    print("\nNotice the values have changed slightly.")
