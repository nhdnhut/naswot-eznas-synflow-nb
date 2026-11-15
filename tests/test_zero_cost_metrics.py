import unittest

import torch
from torch import nn

from src.zero_cost_metrics import (
    naswot_score,
    synflow_iterative_prune,
    synflow_score,
)


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ZeroCostMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TinyCNN()
        self.inputs = torch.randn(8, 3, 16, 16)
        self.initial_state = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }

    def assert_model_unchanged(self) -> None:
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, self.initial_state[name]), name)

    def test_naswot_returns_symmetric_kernel_and_restores_mode(self) -> None:
        self.model.train()
        result = naswot_score(self.model, self.inputs, jitter=1e-6, return_kernel=True)
        kernel = result["kernel"]
        self.assertEqual(tuple(kernel.shape), (8, 8))
        self.assertTrue(torch.allclose(kernel, kernel.T))
        self.assertTrue(torch.allclose(kernel.diag(), torch.full((8,), result["num_relu_units"], dtype=kernel.dtype)))
        self.assertTrue(torch.isfinite(torch.tensor(result["score"])))
        self.assertTrue(self.model.training)
        self.assert_model_unchanged()

    def test_synflow_score_is_finite_and_does_not_mutate_model(self) -> None:
        result = synflow_score(self.model, (1, 3, 16, 16), device="cpu")
        self.assertGreater(result["score"], 0)
        self.assertGreater(result["scored_parameter_count"], 0)
        self.assert_model_unchanged()

    def test_iterative_pruning_returns_masks_and_does_not_mutate_input(self) -> None:
        result = synflow_iterative_prune(
            self.model,
            (1, 3, 16, 16),
            compression_ratio=2.0,
            steps=4,
            device="cpu",
        )
        self.assertEqual(len(result["history"]), 4)
        self.assertGreaterEqual(result["sparsity"], 0.0)
        self.assertLessEqual(result["sparsity"], 1.0)
        self.assertGreater(result["remaining_parameters"], 0)
        self.assertIsInstance(result["model"], nn.Module)
        self.assert_model_unchanged()


if __name__ == "__main__":
    unittest.main()
