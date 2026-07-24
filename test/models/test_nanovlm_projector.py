import unittest

import torch
from torch import nn
from transformers import PretrainedConfig

from lmms_engine.models.nanovlm import (
    NanovlmConfig,
    NanovlmSwiGLUProjector,
    build_nanovlm_projector,
)


class TestNanovlmProjector(unittest.TestCase):
    def test_legacy_config_defaults_to_two_layer_gelu_mlp(self):
        config = NanovlmConfig(
            vision_config=PretrainedConfig(hidden_size=8),
            text_config=PretrainedConfig(hidden_size=6, vocab_size=32),
        )

        self.assertEqual(config.projector_type, "mlp")
        self.assertEqual(config.projector_num_layers, 2)
        self.assertEqual(config.projector_hidden_act, "gelu")
        self.assertTrue(config.projector_bias)

        projector = build_nanovlm_projector(
            projector_type=config.projector_type,
            in_dim=config.vision_feature_dim,
            out_dim=config.text_config.hidden_size,
            hidden_dim=config.projector_hidden_size,
            num_layers=config.projector_num_layers,
            act_name=config.projector_hidden_act,
            bias=config.projector_bias,
        )

        self.assertIsInstance(projector, nn.Sequential)
        self.assertIsInstance(projector[0], nn.Linear)
        self.assertIsInstance(projector[1], nn.GELU)
        self.assertIsInstance(projector[2], nn.Linear)
        self.assertEqual(
            list(projector.state_dict()),
            ["0.weight", "0.bias", "2.weight", "2.bias"],
        )

    def test_linear_projector_shape_and_bias(self):
        projector = build_nanovlm_projector(
            projector_type="linear",
            in_dim=8,
            out_dim=6,
            bias=False,
        )
        inputs = torch.randn(2, 5, 8)

        self.assertIsInstance(projector, nn.Linear)
        self.assertIsNone(projector.bias)
        self.assertEqual(projector(inputs).shape, (2, 5, 6))

    def test_mlp_num_layers_is_honored(self):
        projector = build_nanovlm_projector(
            projector_type="mlp",
            in_dim=8,
            out_dim=6,
            hidden_dim=10,
            num_layers=4,
            act_name="silu",
        )

        linear_layers = [module for module in projector.modules() if isinstance(module, nn.Linear)]
        silu_layers = [module for module in projector.modules() if isinstance(module, nn.SiLU)]

        self.assertEqual(len(linear_layers), 4)
        self.assertEqual(len(silu_layers), 3)
        self.assertEqual(linear_layers[0].in_features, 8)
        self.assertEqual(linear_layers[-1].out_features, 6)
        self.assertEqual(projector(torch.randn(3, 8)).shape, (3, 6))

    def test_legacy_mlp_one_layer_is_linear(self):
        projector = build_nanovlm_projector(
            projector_type="mlp",
            in_dim=8,
            out_dim=6,
            num_layers=1,
        )
        self.assertIsInstance(projector, nn.Linear)

    def test_swiglu_forward_backward_and_parameter_count(self):
        in_dim = 8
        hidden_dim = 10
        out_dim = 6
        projector = build_nanovlm_projector(
            projector_type="swiglu",
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
        )
        inputs = torch.randn(2, 5, in_dim, requires_grad=True)
        outputs = projector(inputs)
        outputs.square().mean().backward()

        self.assertIsInstance(projector, NanovlmSwiGLUProjector)
        self.assertEqual(projector.gate_up_proj.out_features, 2 * hidden_dim)
        self.assertEqual(projector.down_proj.in_features, hidden_dim)
        self.assertEqual(outputs.shape, (2, 5, out_dim))
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        for parameter in projector.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

        expected_parameter_count = (
            in_dim * 2 * hidden_dim
            + 2 * hidden_dim
            + hidden_dim * out_dim
            + out_dim
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in projector.parameters()),
            expected_parameter_count,
        )

    def test_projector_state_dict_round_trip(self):
        torch.manual_seed(7)
        projector = build_nanovlm_projector(
            projector_type="mlp",
            in_dim=8,
            out_dim=6,
            hidden_dim=10,
            num_layers=3,
            act_name="gelu",
        )
        clone = build_nanovlm_projector(
            projector_type="mlp",
            in_dim=8,
            out_dim=6,
            hidden_dim=10,
            num_layers=3,
            act_name="gelu",
        )
        clone.load_state_dict(projector.state_dict(), strict=True)

        inputs = torch.randn(2, 4, 8)
        torch.testing.assert_close(projector(inputs), clone(inputs))

    def test_invalid_projector_arguments_raise(self):
        with self.assertRaisesRegex(ValueError, "Unsupported projector_type"):
            build_nanovlm_projector("unknown", in_dim=8, out_dim=6)
        with self.assertRaisesRegex(ValueError, "num_layers"):
            build_nanovlm_projector("mlp", in_dim=8, out_dim=6, num_layers=0)
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            build_nanovlm_projector("swiglu", in_dim=8, out_dim=6, hidden_dim=-1)

        with self.assertRaisesRegex(ValueError, "Unsupported projector_type"):
            NanovlmConfig(
                vision_config=PretrainedConfig(hidden_size=8),
                text_config=PretrainedConfig(hidden_size=6),
                projector_type="unknown",
            )


if __name__ == "__main__":
    unittest.main()
