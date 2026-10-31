from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import chock_surgeon.live as live_module
from chock_surgeon.trajectory import FlatTrajectory


class VerlContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_registered_live_loop_returns_current_agent_loop_output(self) -> None:
        registry = {}

        class AgentLoopBase:
            def __init__(self, *, server_manager, tokenizer, **kwargs):
                del kwargs
                self.server_manager = server_manager
                self.tokenizer = tokenizer
                self.continuous_token_builder = object()
                self.rollout_config = SimpleNamespace(prompt_length=64, response_length=64)

        class AgentLoopMetrics:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class AgentLoopOutput:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        def register(name):
            def decorator(value):
                registry[name] = value
                return value

            return decorator

        verl = ModuleType("verl")
        experimental = ModuleType("verl.experimental")
        agent_loop_package = ModuleType("verl.experimental.agent_loop")
        agent_loop = ModuleType("verl.experimental.agent_loop.agent_loop")
        agent_loop.AgentLoopBase = AgentLoopBase
        agent_loop.AgentLoopMetrics = AgentLoopMetrics
        agent_loop.AgentLoopOutput = AgentLoopOutput
        agent_loop.register = register
        fake_modules = {
            "verl": verl,
            "verl.experimental": experimental,
            "verl.experimental.agent_loop": agent_loop_package,
            "verl.experimental.agent_loop.agent_loop": agent_loop,
        }

        try:
            with patch.dict(sys.modules, fake_modules):
                loaded = importlib.reload(live_module)
                self.assertIs(loaded.LiveChockAgentLoop, registry["chock_live"])
                calls = {}

                class Bridge:
                    def __init__(self, *args, **kwargs):
                        calls["bridge_args"] = args
                        calls["bridge_kwargs"] = kwargs

                    async def start(self):
                        return "http://127.0.0.1:1234/v1"

                    async def close(self):
                        calls["closed"] = True

                async def run_chock_eval(**kwargs):
                    calls["run"] = kwargs
                    flat = FlatTrajectory(
                        prompt_ids=(1, 2),
                        response_ids=(3, 4),
                        response_mask=(1, 0),
                        old_logprobs=(-0.1, 0.0),
                        reward=0.75,
                        num_turns=2,
                        metadata={
                            "provenance": {
                                "checkpoint": "Qwen/test@revision",
                                "intervention_sha256": "a" * 64,
                            }
                        },
                    )
                    return SimpleNamespace(flat=flat, elapsed_seconds=1.5, exit_code=0)

                loaded.VerlOpenAIBridge = Bridge
                loaded.run_chock_eval = run_chock_eval
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    spec = root / "case.zon"
                    spec.write_text(".{}")
                    project = root / "project"
                    project.mkdir()
                    output_dir = root / "rollouts"
                    loop = loaded.LiveChockAgentLoop(
                        trainer_config=object(),
                        server_manager=object(),
                        tokenizer=object(),
                        processor=None,
                        dataset_cls=object,
                        data_config=object(),
                        model="Qwen/test",
                    )
                    output = await loop.run(
                        {"temperature": 0.8, "seed": 17},
                        chock_spec=str(spec),
                        chock_project=str(project),
                        chock_output_dir=str(output_dir),
                        chock_checkpoint="Qwen/test@revision",
                        chock_intervention="persona-v1",
                        chock_intervention_sha256="a" * 64,
                        extra_info={"group": 2, "rollout": 1},
                    )

                self.assertEqual([1, 2], output.prompt_ids)
                self.assertEqual([3, 4], output.response_ids)
                self.assertEqual([1, 0], output.response_mask)
                self.assertEqual([-0.1, 0.0], output.response_logprobs)
                self.assertEqual(0.75, output.reward_score)
                self.assertEqual(2, output.num_turns)
                self.assertTrue(calls["closed"])
                self.assertIs(
                    loop.continuous_token_builder,
                    calls["bridge_kwargs"]["continuous_token_builder"],
                )
                self.assertEqual(
                    "a" * 64,
                    calls["run"]["provenance"]["intervention_sha256"],
                )
                self.assertEqual(17, calls["run"]["provenance"]["seed"])
        finally:
            importlib.reload(live_module)


if __name__ == "__main__":
    unittest.main()
