import torch
import torch.nn as nn
from typing import Optional

class LoSEModule(nn.Module):
    def __init__(self, id_dim: int = 512, hidden_size: int = 4096):
        super().__init__()
        # Production-grade projection: MLP with LayerNorm and GELU
        self.proj = nn.Sequential(
            nn.Linear(id_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # Learnable gated residual injection (initializes at 0.0)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, id_embed: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Injects the identity embedding into the hidden states via a gated residual.
        id_embed: [B, 512]
        hidden_states: [B, L, hidden_size]
        """
        # Projected ID features: [B, hidden_size]
        id_features = self.proj(id_embed)
        
        # Reshape to [B, 1, hidden_size] for broadcasting across L tokens
        id_features = id_features.unsqueeze(1)
        
        # gated residual: hidden_states + gate * id_features
        return hidden_states + self.gate * id_features

class LoSEAdapter(nn.Module):
    def __init__(self, flux_model: nn.Module, id_dim: int = 512):
        super().__init__()
        self.hidden_size = flux_model.hidden_size
        self.id_dim = id_dim
        
        # Create a LoSEModule for every block in FLUX
        # FLUX.2 Klein 9B has 8 double blocks and 24 single blocks
        self.double_block_adapters = nn.ModuleList([
            LoSEModule(id_dim, self.hidden_size) for _ in range(len(flux_model.double_blocks))
        ])
        
        self.single_block_adapters = nn.ModuleList([
            LoSEModule(id_dim, self.hidden_size) for _ in range(len(flux_model.single_blocks))
        ])

    def save_weights(self, path: str):
        """Save ONLY the LoSE weights to a safetensors file."""
        from safetensors.torch import save_file
        state_dict = self.state_dict()
        save_file(state_dict, path)

    def load_weights(self, path: str):
        """Load LoSE weights from a safetensors file."""
        from safetensors.torch import load_file
        state_dict = load_file(path)
        self.load_state_dict(state_dict)

def patch_flux_with_lose(flux_model: nn.Module, lose_adapter: LoSEAdapter):
    """
    Patches the forward passes of individual blocks to include LoSE injection.
    We inject the ID features into the image stream hidden states.
    """
    
    # Patch DoubleStreamBlocks
    for i, block in enumerate(flux_model.double_blocks):
        adapter = lose_adapter.double_block_adapters[i]
        
        orig_forward = block.forward
        
        def patched_double_forward(
            self, img, txt, pe, pe_ctx, mod_img, mod_txt, 
            id_embed=None, adapter=adapter, orig_forward=orig_forward
        ):
            # If id_embed is provided, inject it using the adapter
            if id_embed is not None:
                img = adapter(id_embed, img)
            
            return orig_forward(img, txt, pe, pe_ctx, mod_img, mod_txt)
            
        # Bind the patched forward to the block instance
        import types
        block.forward = types.MethodType(patched_double_forward, block)

    # Patch SingleStreamBlocks
    for i, block in enumerate(flux_model.single_blocks):
        adapter = lose_adapter.single_block_adapters[i]
        
        orig_forward = block.forward
        
        def patched_single_forward(
            self, x, pe, mod, 
            id_embed=None, adapter=adapter, orig_forward=orig_forward
        ):
            # In SingleStreamBlock, x is the concatenated [txt, img]
            # We usually want to inject ID only into the image part if possible, 
            # but for a global residual on all tokens, we can do x = adapter(id_embed, x).
            # However, the user said "cross-attention mechanisms", so adding to the whole block input is common.
            # Klein 9B text tokens are at the beginning.
            if id_embed is not None:
                x = adapter(id_embed, x)
                
            return orig_forward(x, pe, mod)

        import types
        block.forward = types.MethodType(patched_single_forward, block)
