"""Command-line inspection for the Chock/Surgeon artifact boundary."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from .trajectory import ChockDataset, ChockTrajectory, ContractError
from .transforms import HereticPlan, TransformError, run_heretic
from .verl_adapter import agent_loop_config_path, dataset_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chock-surgeon")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="verify a Chock RL dataset and all selected artifacts")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--split", choices=("train", "validation", "test"))
    inspect = commands.add_parser("inspect", help="verify and summarize one Chock trajectory")
    inspect.add_argument("trajectory", type=Path)
    export = commands.add_parser("export-verl", help="write verified captured-rollout rows for veRL")
    export.add_argument("dataset", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--split", choices=("train", "validation", "test"), default="train")
    heretic = commands.add_parser("heretic", help="run a pinned Heretic weight transform")
    heretic.add_argument("plan", type=Path)
    heretic.add_argument("config", type=Path)
    heretic.add_argument("output", type=Path)
    heretic.add_argument("--executable", default="heretic")
    heretic.add_argument("--timeout", type=float, default=24 * 60 * 60)
    commands.add_parser("verl-config", help="print the installed veRL agent-loop configuration path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            dataset = ChockDataset.load(args.dataset, split=args.split)
            summary = {
                "schema": "chock-surgeon.validation",
                "version": 1,
                "dataset": str(dataset.path),
                "entries": len(dataset.entries),
                "groups": len({entry.group_key for entry in dataset.entries}),
                "split": args.split,
            }
        elif args.command == "inspect":
            trajectory = ChockTrajectory.load(args.trajectory)
            flat = trajectory.flatten()
            summary = {
                "schema": "chock-surgeon.trajectory",
                "version": 1,
                "trajectory": str(trajectory.path),
                "turns": flat.num_turns,
                "prompt_tokens": len(flat.prompt_ids),
                "response_tokens": len(flat.response_ids),
                "trainable_tokens": sum(flat.response_mask),
                "reward": flat.reward,
                "metadata": flat.metadata,
            }
        elif args.command == "export-verl":
            dataset = ChockDataset.load(args.dataset, split=args.split)
            with args.output.open("x", encoding="utf-8") as stream:
                for row in dataset_rows(dataset.entries):
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    stream.write("\n")
            summary = {
                "schema": "chock-surgeon.verl-export",
                "version": 1,
                "dataset": str(dataset.path),
                "output": str(args.output),
                "entries": len(dataset.entries),
                "split": args.split,
                "agent_name": "chock_captured",
            }
        elif args.command == "heretic":
            result = asyncio.run(
                run_heretic(
                    HereticPlan.load(args.plan),
                    config=args.config,
                    output=args.output,
                    executable=args.executable,
                    timeout_seconds=args.timeout,
                )
            )
            summary = {
                "schema": "chock-surgeon.transform-run",
                "version": 1,
                "kind": "heretic",
                "output": str(result.output),
                "audit": str(result.audit),
                "manifest": str(result.manifest),
                "manifest_sha256": result.manifest_sha256,
                "elapsed_seconds": result.elapsed_seconds,
            }
        else:
            print(agent_loop_config_path())
            return 0
    except (ContractError, TransformError) as error:
        print(f"chock-surgeon: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
