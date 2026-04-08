import torch
import sys
import os
from unittest import TestCase, main

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from toolkit.models.lose_adapter import LoSEModule, LoSEAdapter

class TestLoSE(TestCase):
    def test_lose_module_shapes(self):
        hidden_size = 4096
        id_dim = 512
        batch_size = 2
        seq_len = 64
        
        module = LoSEModule(id_dim=id_dim, hidden_size=hidden_size)
        
        id_embed = torch.randn(batch_size, id_dim)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        output = module(id_embed, hidden_states)
        
        self.assertEqual(output.shape, hidden_states.shape, "Output shape should match input hidden states shape")
        self.assertTrue(torch.allclose(output, hidden_states), "Output should be identical to input when gate is 0.0")
        
        # Test grad flow
        output.sum().backward()
        self.assertIsNotNone(module.proj[0].weight.grad, "Gradient should flow to proj weights")
        self.assertIsNotNone(module.gate.grad, "Gradient should flow to gate parameter")

    def test_lose_adapter_initialization(self):
        # Mock a minimal FLUX model structure
        class MockBlock:
            def __init__(self):
                self.forward_called = False
            def forward(self, *args, **kwargs):
                self.forward_called = True
                return args[:2] if len(args) > 1 else args[0]

        class MockFlux:
            def __init__(self):
                self.hidden_size = 4096
                self.double_blocks = [MockBlock() for _ in range(8)]
                self.single_blocks = [MockBlock() for _ in range(24)]

        flux_model = MockFlux()
        adapter = LoSEAdapter(flux_model)
        
        self.assertEqual(len(adapter.double_block_adapters), 8)
        self.assertEqual(len(adapter.single_block_adapters), 24)
        
        # Check that gate is initialized to 0
        self.assertEqual(adapter.double_block_adapters[0].gate.item(), 0.0)

if __name__ == '__main__':
    # Run tests manually
    suite = TestLoSE()
    suite.test_lose_module_shapes()
    suite.test_lose_adapter_initialization()
    print("All LoSE unit tests passed!")
