"""An OpenAI-compatible Chock bridge over veRL's token rollout client."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4


class BridgeError(ValueError):
    """A request or model response cannot be represented losslessly."""


@dataclass(frozen=True)
class ParsedAssistant:
    content: str
    reasoning: str
    tool_calls: tuple[Mapping[str, Any], ...]


_tool_call = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_thinking = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


def parse_qwen_assistant(text: str) -> ParsedAssistant:
    """Parse Qwen's documented tool-call tags without changing their JSON."""

    reasoning_parts = _thinking.findall(text)
    visible = _thinking.sub("", text)
    calls: list[Mapping[str, Any]] = []
    for index, matched in enumerate(_tool_call.findall(visible)):
        try:
            value = json.loads(matched)
        except json.JSONDecodeError as error:
            raise BridgeError(f"Qwen tool call {index} is not JSON: {error}") from error
        if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not value["name"]:
            raise BridgeError(f"Qwen tool call {index} needs a non-empty string name")
        arguments = value.get("arguments", {})
        if isinstance(arguments, str):
            try:
                json.loads(arguments)
            except json.JSONDecodeError as error:
                raise BridgeError(f"Qwen tool call {index} arguments are not JSON: {error}") from error
            encoded_arguments = arguments
        else:
            try:
                encoded_arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            except (TypeError, ValueError) as error:
                raise BridgeError(f"Qwen tool call {index} arguments are not JSON values") from error
        calls.append({"name": value["name"], "arguments": encoded_arguments})
    return ParsedAssistant(
        content=_tool_call.sub("", visible),
        reasoning="\n".join(reasoning_parts),
        tool_calls=tuple(calls),
    )


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(isinstance(one, bool) or not isinstance(one, int) or one < 0 for one in value):
        raise BridgeError("the tokenizer did not return a flat list of non-negative token IDs")
    return value


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _sse(data: Mapping[str, Any], *, event: str | None = None) -> bytes:
    prefix = b"" if event is None else f"event: {event}\n".encode()
    return prefix + b"data: " + _json(data) + b"\n\n"


def _decode_visible(tokenizer: Any, response_ids: list[int]) -> str:
    try:
        text = tokenizer.decode(response_ids, skip_special_tokens=False)
    except Exception as error:
        raise BridgeError(f"the tokenizer could not decode veRL's response: {error}") from error
    eos = getattr(tokenizer, "eos_token", None)
    if isinstance(eos, str) and eos:
        while text.endswith(eos):
            text = text[: -len(eos)]
    return text


class VerlOpenAIBridge:
    """Serve the subset of OpenAI chat completions Chock uses.

    Requests are serialized per bridge because one Chock session issues model
    turns sequentially. Different rollouts use separate bridge instances and
    veRL's server manager still load-balances them across rollout replicas.
    """

    def __init__(
        self,
        server_manager: Any,
        tokenizer: Any,
        sampling_params: Mapping[str, Any],
        *,
        model: str,
        parser: str = "qwen",
        request_namespace: str = "",
        continuous_token_builder: Any | None = None,
        max_request_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if parser != "qwen":
            raise BridgeError(f"unsupported model parser {parser!r}")
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.sampling_params = dict(sampling_params)
        self.model = model
        self.request_namespace = request_namespace or uuid4().hex
        self.continuous_token_builder = continuous_token_builder
        self.max_request_bytes = max_request_bytes
        self._server: asyncio.AbstractServer | None = None
        self._turn = 0
        self._lock = asyncio.Lock()
        self._request_messages: list[Mapping[str, Any]] | None = None
        self._runtime_ids: list[int] = []

    async def start(self) -> str:
        if self._server is not None:
            raise RuntimeError("the bridge is already running")
        self._server = await asyncio.start_server(self._handle_connection, "127.0.0.1", 0)
        address = self._server.sockets[0].getsockname()
        return f"http://127.0.0.1:{address[1]}/v1"

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def __aenter__(self) -> "VerlOpenAIBridge":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def complete(self, request: Mapping[str, Any]) -> bytes:
        messages = request.get("messages")
        tools = request.get("tools")
        if not isinstance(messages, list):
            raise BridgeError("messages must be an array")
        if tools is not None and not isinstance(tools, list):
            raise BridgeError("tools must be an array")
        request_model = request.get("model")
        if request_model is not None and request_model != self.model:
            raise BridgeError(
                f"Chock requested model {request_model!r}, but this rollout serves {self.model!r}"
            )

        sampling = dict(self.sampling_params)
        for wire, internal in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("max_tokens", "max_tokens"),
            ("seed", "seed"),
        ):
            if wire in request:
                sampling[internal] = request[wire]
        sampling["logprobs"] = True

        async with self._lock:
            prompt_ids = self._build_prompt(messages, tools)
            turn = self._turn
            self._turn += 1
            namespace = hashlib.sha256(self.request_namespace.encode()).hexdigest()[:16]
            output = await self.server_manager.generate(
                request_id=f"chock-{namespace}",
                prompt_ids=prompt_ids,
                sampling_params=sampling,
            )
            response_ids = _token_ids(output.token_ids)
            if output.log_probs is None or len(output.log_probs) != len(response_ids):
                raise BridgeError("veRL did not return one rollout log-probability per response token")
            old_logprobs = [float(one) for one in output.log_probs]
            if any(not (-float("inf") < one < float("inf")) for one in old_logprobs):
                raise BridgeError("veRL returned a non-finite rollout log-probability")
            text = _decode_visible(self.tokenizer, response_ids)
            parsed = parse_qwen_assistant(text)
            if self.continuous_token_builder is None:
                self._runtime_ids = prompt_ids + response_ids
            else:
                merged = self.continuous_token_builder.merge_assistant_tokens(prompt_ids, response_ids)
                self._runtime_ids = _token_ids(merged.token_ids)
            self._request_messages = [dict(message) for message in messages]
        completion_key = f"{namespace}:{turn}:{response_ids}".encode()
        completion_id = "chatcmpl-chock-" + hashlib.sha256(completion_key).hexdigest()[:20]

        body = bytearray()
        delta: dict[str, Any] = {}
        if parsed.content:
            delta["content"] = parsed.content
        if parsed.reasoning:
            delta["reasoning_content"] = parsed.reasoning
        if parsed.tool_calls:
            delta["tool_calls"] = [
                {
                    "index": index,
                    "id": f"call_{completion_id[-12:]}_{index}",
                    "type": "function",
                    "function": dict(call),
                }
                for index, call in enumerate(parsed.tool_calls)
            ]
        if delta:
            body += _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "model": self.model,
                    "choices": [{"index": 0, "finish_reason": None, "delta": delta}],
                }
            )
        body += _sse(
            {
                "prompt_token_ids": prompt_ids,
                "response_token_ids": response_ids,
                "response_mask": [1] * len(response_ids),
                "old_logprobs": old_logprobs,
            },
            event="chock.training_trace.v1",
        )
        body += _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if parsed.tool_calls else "stop",
                        "delta": {},
                    }
                ],
            }
        )
        body += b"data: [DONE]\n\n"
        return bytes(body)

    def _build_prompt(self, messages: list[Any], tools: list[Any] | None) -> list[int]:
        if self.continuous_token_builder is None:
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                )
            except Exception as error:
                raise BridgeError(f"the tokenizer could not encode the Chock request: {error}") from error
            return _token_ids(encoded)

        if self._request_messages is None:
            try:
                encoded = self.continuous_token_builder.build_initial_tokens(messages, tools=tools)
            except Exception as error:
                raise BridgeError(f"continuous-token initial prompt construction failed: {error}") from error
            return _token_ids(encoded)

        previous_count = len(self._request_messages)
        if messages[:previous_count] != self._request_messages:
            raise BridgeError("Chock rewrote an earlier message in a continuous-token session")
        if len(messages) <= previous_count or not isinstance(messages[previous_count], dict):
            raise BridgeError("Chock did not append the prior assistant message")
        if messages[previous_count].get("role") != "assistant":
            raise BridgeError("the message after the prior request is not the assistant response")
        previous_messages = messages[: previous_count + 1]
        if len(messages) == len(previous_messages):
            raise BridgeError("a new model turn has no appended environment or user context")
        try:
            merged = self.continuous_token_builder.merge_context_tokens(
                previous_messages,
                messages,
                self._runtime_ids,
                tools=tools,
            )
        except Exception as error:
            raise BridgeError(f"continuous-token context merge failed: {error}") from error
        return _token_ids(merged.token_ids)

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status = "200 OK"
        content_type = "application/json"
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("ascii").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {}
            for line in lines[1:]:
                if not line:
                    continue
                name, value = line.split(":", 1)
                headers[name.lower()] = value.strip()
            length = int(headers.get("content-length", "0"))
            if length < 0 or length > self.max_request_bytes:
                raise BridgeError("request body is too large")
            body = await reader.readexactly(length)
            if method == "GET" and target.rstrip("/") == "/v1/models":
                response = _json({"object": "list", "data": [{"id": self.model, "object": "model"}]})
            elif method == "POST" and target.rstrip("/") == "/v1/chat/completions":
                request = json.loads(body)
                response = await self.complete(_object_request(request))
                content_type = "text/event-stream"
            else:
                status = "404 Not Found"
                response = _json({"error": {"message": "unknown bridge endpoint"}})
        except (BridgeError, ValueError, json.JSONDecodeError, asyncio.IncompleteReadError) as error:
            status = "400 Bad Request"
            response = _json({"error": {"message": str(error)}})
        except Exception as error:
            status = "500 Internal Server Error"
            response = _json({"error": {"message": f"rollout bridge failed: {error}"}})
        writer.write(
            f"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {len(response)}\r\nConnection: close\r\n\r\n".encode()
            + response
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()


def _object_request(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BridgeError("request body must be a JSON object")
    return value
