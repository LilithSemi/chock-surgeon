from __future__ import annotations

import unittest
from types import SimpleNamespace

from chock_surgeon.steering import (
    SteeringError,
    SteeringProfile,
    _qwen_layers,
    steering_coefficient,
)


def profile_value() -> dict:
    return {
        "schema": "chock-surgeon.steering",
        "version": 1,
        "checkpoint": "Qwen/Qwen3-8B",
        "revision": "0123456789abcdef",
        "model_family": "qwen3",
        "num_layers": 36,
        "intermediate_size": 12288,
        "hook": "mlp.down_proj.input",
        "normalization": "median_residual_l2",
        "token_scope": "all",
        "interventions": [
            {"layer": 14, "neuron": 8949, "dose": 1.0},
            {"layer": 13, "neuron": 7956, "dose": -1.0},
        ],
    }


class SteeringProfileTests(unittest.TestCase):
    def test_qwen_layers_resolves_raw_and_wrapped_models(self) -> None:
        layers = [object(), object()]
        raw = SimpleNamespace(model=SimpleNamespace(layers=layers))
        wrapped = SimpleNamespace(model=raw)
        peft = SimpleNamespace(get_base_model=lambda: raw, model=wrapped)
        self.assertIs(layers, _qwen_layers(raw))
        self.assertIs(layers, _qwen_layers(wrapped))
        self.assertIs(layers, _qwen_layers(peft))

    def test_qwen_layers_rejects_unknown_layout(self) -> None:
        with self.assertRaisesRegex(SteeringError, "decoder layers"):
            _qwen_layers(SimpleNamespace())

    def test_neuron_persona_normalization_formula(self) -> None:
        self.assertEqual(6.0, steering_coefficient(1.5, 8.0, 2.0))
        with self.assertRaisesRegex(SteeringError, "direction_norm must be positive"):
            steering_coefficient(1.0, 8.0, 0.0)

    def test_profile_hash_is_canonical_and_covers_every_intervention(self) -> None:
        value = profile_value()
        first = SteeringProfile.parse(value)
        reordered = {key: value[key] for key in reversed(value)}
        second = SteeringProfile.parse(reordered)
        self.assertEqual(first.sha256, second.sha256)
        changed = profile_value()
        changed["interventions"][0]["dose"] = 0.75
        self.assertNotEqual(first.sha256, SteeringProfile.parse(changed).sha256)

    def test_profile_rejects_coordinates_from_another_architecture(self) -> None:
        value = profile_value()
        value["interventions"][0]["neuron"] = value["intermediate_size"]
        with self.assertRaisesRegex(SteeringError, "outside"):
            SteeringProfile.parse(value)

    def test_profile_rejects_duplicate_neuron_pushes(self) -> None:
        value = profile_value()
        value["interventions"].append(dict(value["interventions"][0]))
        with self.assertRaisesRegex(SteeringError, "more than once"):
            SteeringProfile.parse(value)


if __name__ == "__main__":
    unittest.main()
