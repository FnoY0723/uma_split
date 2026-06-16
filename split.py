import torch
import torch.nn as nn
from typing import Tuple, Optional

class Split(nn.Module):
    """
    Split module.
    
    This module predicts new representations by extracting split features, increasing sequence 
    length by seamlessly interleaving the predicted sequences. Designed for Non-Autoregressive ASR.
    Part of "UMA-SPLIT: UNIMODAL AGGREGATION FOR BOTH ENGLISH AND MANDARIN NON-AUTOREGRESSIVE SPEECH RECOGNITION".
    """
    def __init__(
        self,
        output_size: int = 256,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self._output_size = output_size

        self.linear_split = nn.Sequential(
            nn.Linear(output_size, output_size * 4),
            nn.SiLU(),  # PyTorch equivalent of Swish
            nn.Dropout(dropout_rate),
            nn.Linear(output_size * 4, output_size * 1),
        )
        self.after_norm = nn.LayerNorm(output_size)
        
    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        decoder_out: torch.Tensor,
        decoder_out_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward function for the Split module.
        
        Args:
            decoder_out (torch.Tensor): Decoder output features. Shape: (Batch, SeqLen, Dim).
            decoder_out_lens (torch.Tensor, optional): Length of sequences. Shape: (Batch,).
            
        Returns:
            torch.Tensor: Split decoded features. Shape: (Batch, SeqLen * 2, Dim).
            torch.Tensor: New sequence length constraints. Shape: (Batch,).
        """
        batch, seq_len, dim = decoder_out.shape
        
        # Branch processing to predict split representation
        new_decoder_out = self.linear_split(decoder_out)
        new_decoder_out = new_decoder_out.reshape(batch, seq_len, 1, dim)
        
        # Concatenate and interleave generated representation with original ones
        new_decoder_out = torch.cat([decoder_out.unsqueeze(2), new_decoder_out], dim=2)
        
        # Flatten temporal sequences -> effectively doubles sequence length
        new_decoder_out = new_decoder_out.reshape(batch, -1, dim)
        new_decoder_out = self.after_norm(new_decoder_out)
        
        if decoder_out_lens is not None:
            # Updating length bounds to align with expanded sequence 
            decoder_out_lens = 2 * decoder_out_lens
        
        return new_decoder_out, decoder_out_lens
