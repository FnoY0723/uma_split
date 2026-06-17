import torch
import torch.nn as nn
from typing import Tuple, Optional

def make_pad_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Create a boolean mask for padded sequence parts.
    
    Args:
        lengths (torch.Tensor): Tensor indicating lengths of the sequences.
        max_len (int, optional): Maximum length of sequences.
        
    Returns:
        torch.Tensor: Boolean mask where True indicates padded positions.
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    seq_range = torch.arange(max_len, device=lengths.device)
    return seq_range.unsqueeze(0) >= lengths.unsqueeze(1)


class UMA(nn.Module):
    """
    UMA (Unimodal Aggregation) module.
    
    This module aggregates frame-level acoustic features into phoneme-level representations 
    by detecting unimodal distributions (valleys) in the predicted connection weights.
    Part of "UMA-SPLIT: UNIMODAL AGGREGATION FOR BOTH ENGLISH AND MANDARIN NON-AUTOREGRESSIVE SPEECH RECOGNITION".
    """
    def __init__(
        self,
        input_size: int = 256,
        output_size: int = 256,
    ):
        super().__init__()
        self._output_size = output_size
        
        self.linear_sigmoid = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.SiLU(),  # PyTorch equivalent of Swish
            nn.Linear(input_size, 1),
            nn.Sigmoid(),
        )

        self.after_norm = nn.LayerNorm(input_size)
        
        # Initialize weights for the pre-sigmoid linear layer properly
        nn.init.xavier_uniform_(self.linear_sigmoid[-2].weight, gain=nn.init.calculate_gain("sigmoid"))
        nn.init.zeros_(self.linear_sigmoid[-2].bias)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (Batch, Length, InputSize).
            ilens (torch.Tensor): Input length tensor of shape (Batch,).
            
        Returns:
            torch.Tensor: Aggregated output tensor of shape (Batch, NewLength, OutputSize).
            torch.Tensor: Output lengths of shape (Batch,).
        """
        input_masks = make_pad_mask(ilens, max_len=xs_pad.size(1)).to(xs_pad.device)  # (Batch, Length)

        batch, length, _ = xs_pad.size()
        uma_weights = self.linear_sigmoid(xs_pad)  # (Batch, Length, 1)
        uma_weights = uma_weights.masked_fill(input_masks.unsqueeze(-1), 0.0)

        # ---------------- Unimodal Detection ---------------- #
        scalar_before = uma_weights[:, :-1, :].detach() # (Batch, Length-1, 1)
        scalar_after = uma_weights[:, 1:, :].detach()   # (Batch, Length-1, 1)
        scalar_before = torch.nn.functional.pad(scalar_before, (0, 0, 1, 0)) # (Batch, Length, 1)
        scalar_after = torch.nn.functional.pad(scalar_after, (0, 0, 0, 1))   # (Batch, Length, 1)

        # Detect valleys
        mask = (uma_weights.lt(scalar_before)) & (uma_weights.lt(scalar_after)) # (Batch, Length, 1)
        mask = mask.reshape(batch, -1) # (Batch, Length)

        mask[:, 0] = True

        # Extract valley start and end points
        batch_index = mask.nonzero()[:, 0]       # (k, 1)
        valley_index_start = mask.nonzero()[:, 1] # (k, 1)
        
        mask[:, 0] = False
        mask[:, -1] = True
        valley_index_end = mask.nonzero()[:, 1] + 2 
        
        # Clip max boundaries to each sample's length (use per-sample ilens)
        # `batch_index` maps each valley to its sample in the batch
        ilens_device = ilens.to(valley_index_end.device)
        max_len_per_valley = ilens_device[batch_index]
        valley_index_end = torch.where(
            valley_index_end > max_len_per_valley,
            max_len_per_valley,
            valley_index_end,
        )

        _, counts = torch.unique(batch_index, return_counts=True) # Number of valleys in each sample
        max_counts = (torch.max(counts)).item() 

        # Create indexing map for processing variably length segmentations
        utri_mat1 = torch.tril(torch.ones(max_counts + 1, max_counts), -1).to(xs_pad.device)
        batch_index_mask = utri_mat1[counts].reshape(-1, 1).nonzero()[:, 0]

        valleys = torch.zeros(batch * max_counts, 2).type_as(valley_index_start)
        valleys[batch_index_mask] = torch.cat((valley_index_start.unsqueeze(1), valley_index_end.unsqueeze(1)), 1)
        
        # Setup aggregation weight mask matching boundaries
        utri_mat = torch.tril(torch.ones(length + 1, length), -1).to(xs_pad.device)
        output_mask = (utri_mat[valleys[:, 1]] - utri_mat[valleys[:, 0]]).reshape(batch, max_counts, length)
        output_mask = output_mask.detach()

        # ---------------- Unimodal Aggregation ---------------- #
        alpha_h = torch.mul(uma_weights, xs_pad)
        
        # Weighted mean aggregation inside boundary
        xs_pad_aggregated = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, uma_weights).clamp_(1e-12)
        xs_pad_aggregated = self.after_norm(xs_pad_aggregated)
        
        olens = counts
        return xs_pad_aggregated, olens
