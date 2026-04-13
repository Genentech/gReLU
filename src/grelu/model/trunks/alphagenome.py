from typing import Optional, Union, Dict, Any
import torch
import torch.nn as nn
from alphagenome_pytorch.model import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy

class AlphaGenomeTrunk(nn.Module):
    """
    A trunk that wraps the AlphaGenome model from alphagenome-pytorch.
    
    Args:
        num_organisms: Number of organisms (default 2: human, mouse).
        organism_index: Default organism index to use for inference.
        output_key: The output modality to extract (e.g., 'atac', 'dnase', 'cage', 'rna_seq', 'contact_maps').
        resolution: The resolution to extract (1 or 128).
        dtype_policy: DtypePolicy for precision control.
        weights_path: Optional path to a pretrained weights file (.pth).
        gradient_checkpointing: If True, enable gradient checkpointing.
        **kwargs: Additional arguments passed to AlphaGenome constructor.
    """
    def __init__(
        self,
        num_organisms: int = 2,
        organism_index: int = 0,
        output_key: str = "atac",
        resolution: int = 128,
        dtype_policy: Optional[DtypePolicy] = None,
        weights_path: Optional[str] = None,
        gradient_checkpointing: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.organism_index = organism_index
        self.output_key = output_key
        self.resolution = resolution
        
        if weights_path:
            self.model = AlphaGenome.from_pretrained(
                weights_path,
                dtype_policy=dtype_policy,
                num_organisms=num_organisms,
                gradient_checkpointing=gradient_checkpointing,
                **kwargs
            )
        else:
            self.model = AlphaGenome(
                num_organisms=num_organisms,
                dtype_policy=dtype_policy,
                gradient_checkpointing=gradient_checkpointing,
                **kwargs
            )
            
        # Determine out_channels for gReLU head
        if output_key in self.model.heads:
            self.out_channels = self.model.heads[output_key].num_tracks
        elif output_key == "contact_maps":
            self.out_channels = 28 # CONTACT_MAPS_OUTPUT_TRACKS
        elif output_key == "splice_sites":
            self.out_channels = 5
        elif output_key == "splice_site_usage":
            self.out_channels = self.model.splice_sites_usage_head.num_output_tracks
        elif output_key == "splice_junctions":
            self.out_channels = self.model.splice_sites_junction_head._num_tissues * 2
        else:
            self.out_channels = 0
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, 4, L)
            
        Returns:
            Output tensor of shape (N, C, L_out)
        """
        # gReLU uses (N, 4, L), AlphaGenome expects (N, L, 4)
        x = x.transpose(1, 2)
        
        # Determine organism index per batch
        batch_size = x.shape[0]
        organism_index = torch.full(
            (batch_size,), 
            self.organism_index, 
            dtype=torch.long, 
            device=x.device
        )
        
        if self.training:
            # Use forward directly during training to keep gradients
            # We set channels_last=False to get (N, C, L) back
            outputs = self.model(x, organism_index, channels_last=False)
        else:
            # Use predict for inference (handles no_grad and autocast)
            outputs = self.model.predict(x, organism_index, channels_last=False)
            
        # Extract desired output
        out = outputs[self.output_key]
        
        if isinstance(out, dict):
            out = out[self.resolution]
            
        # For contact_maps, AlphaGenome returns (B, T, S1, S2) if channels_last=False
        # gReLU expects (B, C, L). For contact maps, this might need special handling if used in 1D tasks.
        # But for standard tracks, it's (B, T, S).
            
        return out
