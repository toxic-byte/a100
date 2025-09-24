import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)列向量
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model//2,)
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用 cos
        
        self.register_buffer('pe', pe)  # 不参与训练，但会随模型保存
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model) + PE
        """
        seq_len = x.size(1)
        pe = self.pe[:seq_len]  # 只取前 seq_len 个位置的编码
        return x + pe.unsqueeze(0)  # (batch_size, seq_len, d_model)
        # unsqueeze(0) 将pe从 (seq_len, d_model) 变为 (1, seq_len, d_model)
        # 这样可以通过广播机制加到 (batch_size, seq_len, d_model) 的输入上