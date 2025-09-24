class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)  # (max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model) + PE
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pe = self.pos_embedding(positions)  # (1, seq_len, d_model)
        return x + pe