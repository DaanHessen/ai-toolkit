import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .iresnet import iresnet100

class FacialIdentityLoss(nn.Module):
    def __init__(self, model_path: str = None, device: str = "cuda"):
        super().__init__()
        # Load the frozen ArcFace model
        self.face_model = iresnet100()
        if model_path:
            # Expecting a standard weight dictionary
            state_dict = torch.load(model_path, map_location="cpu")
            self.face_model.load_state_dict(state_dict)
        
        self.face_model.to(device)
        self.face_model.eval()
        self.face_model.requires_grad_(False)
        
    @staticmethod
    def decode_latents_differentiable(vae, latents, scaling_factor=0.3611, shift_factor=0.1159):
        """
        Differentiable decoding of FLUX latents.
        Uses gradient checkpointing to save VRAM.
        """
        # Correctly unscale: (z_0 - shift) / scale
        # Wait, the user said unscale BEFORE passing to VAE.
        # Diffusers typically does: latents / scaling_factor + shift_factor
        # Let's use the standard Diffusers unscaling for FLUX.
        latents = (latents / scaling_factor) + shift_factor
        
        def _decode(l):
            # l expected to be [B, C, H, W]
            return vae.decode(l, return_dict=False)[0]

        # Use gradient checkpointing on the expensive VAE decoding
        pixels = checkpoint(_decode, latents, use_reentrant=False)
        
        # pixels: [B, 3, H, W], usually in range [-1, 1]
        # Normalize to [0, 1] for general use
        pixels = (pixels + 1.0) / 2.0
        pixels = torch.clamp(pixels, 0.0, 1.0)
        return pixels

    def forward(self, gen_pixels, ref_embed):
        """
        gen_pixels: [B, 3, 512, 512], range [0, 1]
        ref_embed: [B, 512], normalized face embedding
        """
        # ArcFace expects 112x112 input
        gen_pixels_112 = F.interpolate(gen_pixels, size=(112, 112), mode='bilinear', align_corners=False)
        
        # ArcFace normalization: (x - 0.5) / 0.5
        gen_pixels_112 = (gen_pixels_112 - 0.5) / 0.5
        
        # Extract generated embedding
        gen_embed = self.face_model(gen_pixels_112)
        gen_embed = F.normalize(gen_embed, p=2, dim=1)
        ref_embed = F.normalize(ref_embed, p=2, dim=1)
        
        # Loss = 1 - Cosine Similarity
        cosine_sim = (gen_embed * ref_embed).sum(dim=1)
        loss = 1.0 - cosine_sim.mean()
        
        return loss
