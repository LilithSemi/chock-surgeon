"""Response-only LoRA distillation for baking runtime behavior into weights."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any


class DistillationError(RuntimeError):
    """A distillation dataset or training run is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DistillationSample:
    messages: tuple[dict[str, str], ...]
    response: str


@dataclass(frozen=True)
class DistillationDataset:
    path: Path
    samples: tuple[DistillationSample, ...]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "DistillationDataset":
        path = path.expanduser().resolve(strict=True)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DistillationError(f"could not read distillation dataset {path}: {error}") from error
        if not isinstance(document, dict):
            raise DistillationError("distillation dataset must be a JSON object")
        if document.get("schema") != "chock-surgeon.distillation" or document.get("version") != 1:
            raise DistillationError("distillation dataset must use chock-surgeon.distillation version 1")
        raw_samples = document.get("samples")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise DistillationError("distillation samples must be a non-empty array")
        samples = []
        for index, raw in enumerate(raw_samples):
            if not isinstance(raw, dict):
                raise DistillationError(f"samples[{index}] must be an object")
            raw_messages = raw.get("messages")
            if not isinstance(raw_messages, list) or not raw_messages:
                raise DistillationError(f"samples[{index}].messages must be a non-empty array")
            messages = []
            for message_index, message in enumerate(raw_messages):
                if not isinstance(message, dict):
                    raise DistillationError(
                        f"samples[{index}].messages[{message_index}] must be an object"
                    )
                role = message.get("role")
                content = message.get("content")
                if role not in ("system", "user", "assistant", "tool"):
                    raise DistillationError(
                        f"samples[{index}].messages[{message_index}].role is unsupported"
                    )
                if not isinstance(content, str) or not content.strip():
                    raise DistillationError(
                        f"samples[{index}].messages[{message_index}].content must not be empty"
                    )
                messages.append({"role": role, "content": content})
            response = raw.get("response")
            if not isinstance(response, str) or not response.strip():
                raise DistillationError(f"samples[{index}].response must not be empty")
            samples.append(DistillationSample(tuple(messages), response.strip()))
        metadata = {key: value for key, value in document.items() if key != "samples"}
        return cls(path=path, samples=tuple(samples), metadata=metadata)


@dataclass(frozen=True)
class DistillationResult:
    output: Path
    result: Path
    dataset_sha256: str
    samples: int


def train_distillation(
    *,
    model_path: Path,
    dataset_path: Path,
    output: Path,
    epochs: int = 2,
    learning_rate: float = 1e-5,
    rank: int = 16,
    alpha: int = 32,
    max_sequence_length: int = 768,
    gradient_accumulation: int = 4,
    seed: int = 47,
) -> DistillationResult:
    """Train a response-only LoRA against captured teacher completions."""

    if epochs <= 0 or rank <= 0 or alpha <= 0 or max_sequence_length <= 0:
        raise DistillationError("epochs, rank, alpha, and max sequence length must be positive")
    if gradient_accumulation <= 0:
        raise DistillationError("gradient accumulation must be positive")
    if learning_rate <= 0:
        raise DistillationError("learning rate must be positive")
    model_path = model_path.expanduser().resolve(strict=True)
    if not model_path.is_dir():
        raise DistillationError(f"model is not a directory: {model_path}")
    dataset = DistillationDataset.load(dataset_path)
    output = output.expanduser().resolve()
    if output.exists():
        raise DistillationError(f"distillation output already exists: {output}")
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise DistillationError(
            "distillation requires torch, transformers, and peft; use the training environment"
        ) from error

    torch.manual_seed(seed)
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=0.0,
    )
    output.mkdir(parents=True)
    trace_path = output / "training-trace.jsonl"
    update = 0
    losses: list[float] = []
    order = list(range(len(dataset.samples)))
    optimizer.zero_grad(set_to_none=True)
    with trace_path.open("x", encoding="utf-8") as trace:
        for epoch in range(epochs):
            random.shuffle(order)
            for position, sample_index in enumerate(order):
                sample = dataset.samples[sample_index]
                rendered = tokenizer.apply_chat_template(
                    list(sample.messages),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt_ids = tokenizer(
                    rendered, add_special_tokens=False, return_tensors="pt"
                ).input_ids[0]
                response_ids = tokenizer(
                    sample.response + tokenizer.eos_token,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids[0]
                if response_ids.numel() >= max_sequence_length:
                    raise DistillationError(
                        f"sample {sample_index} response exceeds the maximum sequence length"
                    )
                prompt_budget = max_sequence_length - response_ids.numel()
                prompt_ids = prompt_ids[-prompt_budget:]
                input_ids = torch.cat((prompt_ids, response_ids)).unsqueeze(0).to("cuda")
                labels = input_ids.clone()
                labels[:, : prompt_ids.numel()] = -100
                model.train()
                loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
                (loss / gradient_accumulation).backward()
                losses.append(float(loss.detach().cpu()))
                at_epoch_end = position + 1 == len(order)
                if (position + 1) % gradient_accumulation == 0 or at_epoch_end:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update += 1
                    record = {
                        "epoch": epoch + 1,
                        "update": update,
                        "sample": sample_index,
                        "loss": float(loss.detach().cpu()),
                        "gradient_norm": float(grad_norm),
                    }
                    trace.write(json.dumps(record, separators=(",", ":")) + "\n")
                    trace.flush()
                    print(json.dumps(record), flush=True)

    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    result_path = output / "distillation-result.json"
    result_document = {
        "schema": "chock-surgeon.distillation-run",
        "version": 1,
        "model": str(model_path),
        "dataset": str(dataset.path),
        "dataset_sha256": _sha256(dataset.path),
        "samples": len(dataset.samples),
        "epochs": epochs,
        "updates": update,
        "learning_rate": learning_rate,
        "rank": rank,
        "alpha": alpha,
        "max_sequence_length": max_sequence_length,
        "gradient_accumulation": gradient_accumulation,
        "seed": seed,
        "mean_loss": sum(losses) / len(losses),
        "final_loss": losses[-1],
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    result_path.write_text(json.dumps(result_document, indent=2) + "\n", encoding="utf-8")
    return DistillationResult(
        output=output,
        result=result_path,
        dataset_sha256=result_document["dataset_sha256"],
        samples=len(dataset.samples),
    )
