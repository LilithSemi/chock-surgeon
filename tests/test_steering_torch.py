from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # torch is the steer extra and the base package omits it
    raise unittest.SkipTest("torch is not in this environment") from None

from chock_surgeon.steering import (
    SteeringError,
    SteeringProfile,
    install_qwen_steering,
    install_steering,
)
from test_steering import HIDDEN, LAYERS, WIDTH, deepseek_value


class FakeMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(HIDDEN, WIDTH, bias=False)
        self.down_proj = nn.Linear(WIDTH, HIDDEN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.up_proj(x)))


class FakeLayer(nn.Module):
    def __init__(self, shared_name: str | None) -> None:
        super().__init__()
        self.mlp = nn.Module()
        if shared_name is None:
            self.mlp.up_proj = nn.Linear(HIDDEN, WIDTH, bias=False)
            self.mlp.down_proj = nn.Linear(WIDTH, HIDDEN, bias=False)
            self.dense = True
        else:
            self.mlp.experts = nn.ModuleList([FakeMlp() for _ in range(4)])
            setattr(self.mlp, shared_name, FakeMlp())
            self.dense = False
        self.shared_name = shared_name

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        if self.dense:
            return (x + self.mlp.down_proj(torch.relu(self.mlp.up_proj(x))),)
        return (x + getattr(self.mlp, self.shared_name)(x) + self.mlp.experts[0](x),)


class FakeStack(nn.Module):
    def __init__(self, shared_name: str | None, count: int = LAYERS) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, HIDDEN)
        self.layers = nn.ModuleList([FakeLayer(shared_name) for _ in range(count)])

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)[0]
        return x


def qwen_fake(count: int = LAYERS) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Module()
    model.model = FakeStack(None, count)
    model.forward = model.model.forward
    return model


def deepseek_fake(shared_name: str = "shared_experts", nest: str = "language_model") -> nn.Module:
    torch.manual_seed(0)
    model = nn.Module()
    stack = FakeStack(shared_name)
    if nest == "language_model":
        model.language_model = stack
        model.forward = stack.forward
    else:
        model.model = stack
        model.forward = stack.forward
    return model


def fake_qwen_value() -> dict:
    value = deepseek_value()
    value["model_family"] = "qwen3"
    return value


IDS = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
MASK = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])


def install(model: nn.Module, value: dict, fn=install_steering):
    return fn(
        model,
        SteeringProfile.parse(value),
        input_ids=IDS,
        attention_mask=MASK,
        loaded_checkpoint=value["checkpoint"],
        loaded_revision=value["revision"],
    )


def median_residual_norm(model: nn.Module, layer: nn.Module) -> float:
    seen = []
    handle = layer.register_forward_hook(lambda _m, _a, out: seen.append(out[0]))
    with torch.no_grad():
        model(IDS, MASK)
    handle.remove()
    norms = seen[0][:, 1:, :].norm(dim=-1)[MASK[:, 1:].bool()]
    return float(norms.sort().values[(norms.numel() - 1) // 2])


class InstallSteeringTests(unittest.TestCase):
    def test_qwen_shaped_model_gets_the_dense_mlp_coefficient(self) -> None:
        model = qwen_fake()
        value = fake_qwen_value()
        residual = median_residual_norm(model, model.model.layers[0])
        column = model.model.layers[0].mlp.down_proj.weight[:, 3]
        expected = 1.5 * residual / float(column.detach().norm())
        with install(model, value, install_qwen_steering) as session:
            self.assertAlmostEqual(expected, session.coefficients[(0, 3)], places=5)
            self.assertEqual(1, len(model.model.layers[0].mlp.down_proj._forward_pre_hooks))
        self.assertEqual(0, len(model.model.layers[0].mlp.down_proj._forward_pre_hooks))

    def test_qwen_wrapper_refuses_a_deepseek_profile(self) -> None:
        with self.assertRaisesRegex(SteeringError, "install_steering"):
            install(deepseek_fake(), deepseek_value(), install_qwen_steering)

    def test_deepseek_resolves_the_shared_expert_never_a_routed_one(self) -> None:
        for name in ("shared_experts", "shared_expert"):
            with self.subTest(name=name):
                model = deepseek_fake(name)
                layer = model.language_model.layers[0]
                shared = getattr(layer.mlp, name).down_proj
                residual = median_residual_norm(model, layer)
                expected = 1.5 * residual / float(shared.weight[:, 3].detach().norm())
                with install(model, deepseek_value()) as session:
                    self.assertAlmostEqual(expected, session.coefficients[(0, 3)], places=5)
                    self.assertEqual(1, len(shared._forward_pre_hooks))
                    for routed in layer.mlp.experts:
                        self.assertEqual(0, len(routed.down_proj._forward_pre_hooks))

    def test_every_nesting_candidate_resolves(self) -> None:
        for path in ("model", "model.language_model", "language_model.model", "language_model"):
            with self.subTest(path=path):
                stack = deepseek_fake().language_model
                holder = nn.Module()
                node = holder
                *parents, last = path.split(".")
                for name in parents:
                    child = nn.Module()
                    setattr(node, name, child)
                    node = child
                setattr(node, last, stack)
                holder.forward = stack.forward
                with install(holder, deepseek_value()):
                    pass

    def test_injection_moves_only_the_targeted_column(self) -> None:
        model = deepseek_fake()
        shared = model.language_model.layers[0].mlp.shared_experts.down_proj
        base = []
        handle = shared.register_forward_pre_hook(lambda _m, args: base.append(args[0].clone()))
        with torch.no_grad():
            model(IDS, MASK)
        handle.remove()
        steered = []
        with install(model, deepseek_value()) as session:
            handle = shared.register_forward_pre_hook(lambda _m, args: steered.append(args[0].clone()))
            with torch.no_grad():
                model(IDS, MASK)
            handle.remove()
            delta = steered[0] - base[0]
            coefficient = session.coefficients[(0, 3)]
        self.assertTrue(torch.allclose(delta[..., 3], torch.full_like(delta[..., 3], coefficient)))
        others = torch.cat([delta[..., :3], delta[..., 4:]], dim=-1)
        self.assertTrue(torch.equal(others, torch.zeros_like(others)))

    def test_quantized_weights_are_refused(self) -> None:
        model = deepseek_fake()
        shared = model.language_model.layers[0].mlp.shared_experts.down_proj
        shared.weight = nn.Parameter(torch.zeros(HIDDEN, WIDTH, dtype=torch.int8), requires_grad=False)
        with self.assertRaisesRegex(SteeringError, "bf16 checkpoint is required"):
            install(model, deepseek_value())
        shared.weight = nn.Parameter(
            torch.ones(HIDDEN, WIDTH).to(torch.float8_e4m3fn), requires_grad=False
        )
        with self.assertRaisesRegex(SteeringError, "bf16 checkpoint is required"):
            install(model, deepseek_value())

    def test_weight_scale_beside_the_weight_is_refused(self) -> None:
        for name in ("weight_scale", "weight_scale_inv"):
            with self.subTest(name=name):
                model = deepseek_fake()
                shared = model.language_model.layers[0].mlp.shared_experts.down_proj
                shared.register_buffer(name, torch.ones(1))
                with self.assertRaisesRegex(SteeringError, "bf16 checkpoint is required"):
                    install(model, deepseek_value())

    def test_layer_count_must_match_the_profile(self) -> None:
        with self.assertRaisesRegex(SteeringError, "profile expects 3 layers, loaded model differs"):
            install(qwen_fake(2), fake_qwen_value(), install_qwen_steering)

    def test_unresolvable_layers_name_every_candidate(self) -> None:
        with self.assertRaises(SteeringError) as caught:
            install(nn.Linear(2, 2), deepseek_value())
        message = str(caught.exception)
        for path in (
            "model.model.layers",
            "model.model.language_model.layers",
            "model.language_model.model.layers",
            "model.language_model.layers",
        ):
            self.assertIn(path, message)

    def test_missing_shared_expert_names_its_candidates(self) -> None:
        model = deepseek_fake()
        del model.language_model.layers[0].mlp.shared_experts
        with self.assertRaisesRegex(SteeringError, "mlp.shared_experts.down_proj, mlp.shared_expert.down_proj"):
            install(model, deepseek_value())

    def test_width_mismatch_is_still_refused(self) -> None:
        value = deepseek_value()
        value["intermediate_size"] = 32
        with self.assertRaisesRegex(SteeringError, "intermediate width 32"):
            install(deepseek_fake(), value)


if __name__ == "__main__":
    unittest.main()
