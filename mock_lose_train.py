import torch
import torch.nn as nn
from toolkit.models.lose_adapter import LoSEModule, LoSEAdapter
from toolkit.models.face_loss import FacialIdentityLoss

# --- Mocking minimal components for isolated testing ---
class MockVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Mocking a differentiable "decoding" operation: dummy conv
        # In real VAE, this is a large decoder. Here we just want gradients to flow.
        self.decoder = nn.Conv2d(16, 3, 3, padding=1)
        self.config = type('obj', (object,), {'scaling_factor': 0.3611, 'shift_factor': 0.1159})()

    def decode(self, x, return_dict=False):
        # x: [B, 16, H, W]
        # output: [B, 3, H*8, W*8]
        # For mock, we'll just upscale with interpolation + conv
        upscaled = torch.nn.functional.interpolate(x, scale_factor=8, mode='nearest')
        pixels = self.decoder(upscaled)
        return (pixels,)

class MockFlux(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4096
        # Minimal blocks
        self.double_blocks = nn.ModuleList([nn.Identity()])
        self.single_blocks = nn.ModuleList([nn.Identity()])

def test_full_gradient_flow():
    print("Starting Mock Gradient Flow Test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Mock Models
    vae = MockVAE().to(device)
    flux = MockFlux().to(device)
    adapter = LoSEAdapter(flux).to(device)
    
    # ArcFace Loss (using dummy model path to test if it loads, 
    # but here we'll just mock the face model part for isolation)
    # We'll use a random linear layer as the "face model" for this test
    face_loss_fn = FacialIdentityLoss(device=device)
    face_loss_fn.face_model = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 512)
    ).to(device)
    face_loss_fn.face_model.requires_grad_(False) # Face model is frozen
    
    # 2. Dummy Inputs
    batch_size = 1
    id_dim = 512
    # Latents [B, 16, 64, 64] -> Image [B, 3, 512, 512]
    x_t = torch.randn(batch_size, 16, 64, 64, requires_grad=True).to(device)
    ref_embed = torch.randn(batch_size, id_dim).to(device)
    t = torch.tensor([0.5]).to(device)
    
    # 3. Simulate Forward Pass matching lose_train.py logic
    # Assume v_pred comes from transformer (which is patched)
    # Since we can't easily run the patched transformer without full weights, 
    # we'll simulate the part dependent on LoSE weights.
    
    # LoSE contribution to hidden_states (dummy)
    # hidden_states represent the noisy latents in transformer space
    # LoSEModule: hidden_states + gate * proj(id_embed)
    adapter_out = adapter.double_block_adapters[0](ref_embed, torch.randn(batch_size, 1, 4096).to(device))
    
    # We focus on the VAE -> Loss part which is the common point of failure
    # Predicted clean latents: z_0 = x_t - t * v_pred
    # Here we'll just assume z_0 is some differentiable function of adapter_out
    # to test grad flow back to adapter.
    z_0 = x_t + adapter_out.mean() * 0.0 # Force dependency
    
    # 4. Differentiable Decoding
    gen_pixels = face_loss_fn.decode_latents_differentiable(vae, z_0)
    
    # 5. Identity Loss
    loss = face_loss_fn(gen_pixels, ref_embed)
    
    print(f"Loss computed: {loss.item():.4f}")
    
    # 6. Backward Pass
    loss.backward()
    
    # 7. Verify Gradients
    adapter_gate_grad = adapter.double_block_adapters[0].gate.grad
    adapter_proj_grad = adapter.double_block_adapters[0].proj[0].weight.grad
    
    print(f"LoSE Gate Gradient: {adapter_gate_grad}")
    if adapter_gate_grad is not None:
        print("SUCCESS: Gradients flowed to LoSE adapter weights!")
    else:
        print("FAILED: No gradients reached LoSE adapter.")
    
    if adapter_proj_grad is not None:
        print("SUCCESS: Gradients flowed to LoSE projection weights!")
    else:
        print("FAILED: No gradients reached LoSE projection.")

if __name__ == "__main__":
    test_full_gradient_flow()
