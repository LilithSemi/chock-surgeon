"""A small OpenAI-compatible inference server for local model releases."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Sequence
import uuid

from .steering import SteeringProfile, install_qwen_steering


class ServerError(ValueError):
    """The local inference server cannot safely satisfy a request."""


_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def parse_qwen_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert Qwen3.5 tool XML into OpenAI function-call records."""

    calls = []
    for match in _TOOL_CALL.finditer(text):
        name = match.group(1).strip()
        if not name:
            raise ServerError("Qwen emitted a tool call with an empty function name")
        arguments: dict[str, Any] = {}
        for parameter in _PARAMETER.finditer(match.group(2)):
            parameter_name = parameter.group(1).strip()
            raw_value = parameter.group(2).strip()
            if not parameter_name:
                raise ServerError("Qwen emitted a tool call with an empty parameter name")
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
            arguments[parameter_name] = value
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }
        )
    content = _TOOL_CALL.sub("", text).strip()
    return content or None, calls


def _messages_for_qwen(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenAI message fields consumed differently by Qwen's template."""

    normalized = []
    for message in messages:
        result = dict(message)
        if message.get("tool_calls"):
            result["tool_calls"] = []
            for call in message["tool_calls"]:
                copied_call = dict(call)
                function = dict(copied_call.get("function", {}))
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as error:
                        raise ServerError(
                            "an assistant tool call contains invalid JSON arguments"
                        ) from error
                if not isinstance(arguments, dict):
                    raise ServerError("Qwen tool-call arguments must decode to an object")
                function["arguments"] = arguments
                copied_call["function"] = function
                result["tool_calls"].append(copied_call)
        normalized.append(result)
    return normalized


@dataclass(frozen=True)
class Generation:
    content: str | None
    tool_calls: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int


class HarukaEngine:
    """One serialized Transformers model with prompt-normalized steering."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        profile: SteeringProfile | None,
        model_name: str,
        persona: str | None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.profile = profile
        self.model_name = model_name
        self.persona = persona
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        *,
        model_path: Path,
        adapter_path: Path | None,
        profile_path: Path | None,
        model_name: str,
        persona_path: Path | None = None,
    ) -> "HarukaEngine":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ServerError("serving requires torch and transformers; use the training environment") from error
        profile = SteeringProfile.load(profile_path) if profile_path else None
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        if adapter_path is None:
            model = base
        else:
            try:
                from peft import PeftModel
            except ImportError as error:
                raise ServerError("serving a PEFT adapter requires peft") from error
            model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
        persona = persona_path.read_text().strip() if persona_path else None
        return cls(
            model=model,
            tokenizer=tokenizer,
            profile=profile,
            model_name=model_name,
            persona=persona,
        )

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
    ) -> Generation:
        try:
            import torch
        except ImportError as error:
            raise ServerError("serving requires PyTorch") from error
        if not messages:
            raise ServerError("messages must not be empty")
        effective_messages = _messages_for_qwen(messages)
        if self.persona:
            if effective_messages and effective_messages[0].get("role") == "system":
                effective_messages[0]["content"] = (
                    self.persona + "\n\n" + str(effective_messages[0].get("content", ""))
                )
            else:
                effective_messages.insert(0, {"role": "system", "content": self.persona})
        template_arguments: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if tools:
            template_arguments["tools"] = tools
        rendered = self.tokenizer.apply_chat_template(
            effective_messages,
            **template_arguments,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to("cuda")
        with self._lock:
            generation_arguments: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_arguments["temperature"] = temperature
            if self.profile is None:
                with torch.inference_mode():
                    generated = self.model.generate(**inputs, **generation_arguments)
            else:
                session = install_qwen_steering(
                    self.model,
                    self.profile,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    loaded_checkpoint=self.profile.checkpoint,
                    loaded_revision=self.profile.revision,
                )
                with session, torch.inference_mode():
                    generated = self.model.generate(**inputs, **generation_arguments)
        response_ids = generated[0, inputs.input_ids.shape[1] :]
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        content, tool_calls = parse_qwen_tool_calls(text)
        return Generation(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=int(inputs.input_ids.shape[1]),
            completion_tokens=int(response_ids.shape[0]),
        )


def _completion_document(engine: HarukaEngine, request: dict[str, Any]) -> dict[str, Any]:
    generation = engine.generate(
        request.get("messages", []),
        tools=request.get("tools"),
        max_tokens=min(max(int(request.get("max_tokens", 512)), 1), 4096),
        temperature=max(float(request.get("temperature", 0.0)), 0.0),
    )
    finish_reason = "tool_calls" if generation.tool_calls else "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine.model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generation.content,
                    **({"tool_calls": generation.tool_calls} if generation.tool_calls else {}),
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "total_tokens": generation.prompt_tokens + generation.completion_tokens,
        },
    }


def _streaming_delta(message: dict[str, Any]) -> dict[str, Any]:
    """Add fields required specifically by Chat Completions streaming."""

    delta = {"role": message["role"], "content": message.get("content")}
    tool_calls = message.get("tool_calls")
    if tool_calls:
        delta["tool_calls"] = [
            {"index": index, **tool_call} for index, tool_call in enumerate(tool_calls)
        ]
    return delta


def create_app(engine: HarukaEngine, *, api_key: str | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as error:
        raise ServerError("serving requires fastapi and uvicorn") from error

    app = FastAPI(title="chock-surgeon", docs_url=None, redoc_url=None)

    def authorize(request: Request) -> None:
        if api_key and request.headers.get("authorization") != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid bearer token")

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        authorize(request)
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": engine.model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "chock-surgeon",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        authorize(request)
        payload = await request.json()
        try:
            completion = _completion_document(engine, payload)
        except (ServerError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if not payload.get("stream", False):
            return JSONResponse(completion)

        async def events():
            choice = completion["choices"][0]
            message = choice["message"]
            first = dict(completion)
            first.pop("usage", None)
            first["object"] = "chat.completion.chunk"
            first["choices"] = [
                {
                    "index": 0,
                    "delta": _streaming_delta(message),
                    "finish_reason": None,
                }
            ]
            final = {
                "id": completion["id"],
                "object": "chat.completion.chunk",
                "created": completion["created"],
                "model": completion["model"],
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}
                ],
                "usage": completion["usage"],
            }
            yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def serve(
    *,
    model_path: Path,
    adapter_path: Path | None,
    profile_path: Path | None,
    model_name: str,
    persona_path: Path | None,
    host: str,
    port: int,
    api_key: str | None,
) -> None:
    if host not in ("127.0.0.1", "::1", "localhost") and not api_key:
        raise ServerError("a non-loopback listener requires --api-key")
    try:
        import uvicorn
    except ImportError as error:
        raise ServerError("serving requires uvicorn") from error
    engine = HarukaEngine.load(
        model_path=model_path,
        adapter_path=adapter_path,
        profile_path=profile_path,
        model_name=model_name,
        persona_path=persona_path,
    )
    uvicorn.run(create_app(engine, api_key=api_key), host=host, port=port, log_level="info")
