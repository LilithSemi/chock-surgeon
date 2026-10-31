from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
import unittest

from chock_surgeon.bridge import BridgeError, VerlOpenAIBridge, parse_qwen_assistant


class FakeTokenizer:
    eos_token = "<|im_end|>"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None
        self.tools = None

    def apply_chat_template(self, messages, *, tools, add_generation_prompt, tokenize):
        self.messages = messages
        self.tools = tools
        assert add_generation_prompt
        assert tokenize
        return [10, 11]

    def decode(self, token_ids, *, skip_special_tokens):
        assert token_ids == [20, 21]
        assert not skip_special_tokens
        return self.response


class FakeManager:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(token_ids=[20, 21], log_probs=[-0.25, -0.5])


class FakeContinuousBuilder:
    def build_initial_tokens(self, messages, *, tools):
        assert messages == [{"role": "user", "content": "inspect"}]
        return [10, 11]

    def merge_assistant_tokens(self, runtime_token_ids, assistant_token_ids):
        return SimpleNamespace(token_ids=list(runtime_token_ids) + list(assistant_token_ids))

    def merge_context_tokens(self, previous_messages, updated_messages, runtime_token_ids, *, tools):
        assert previous_messages[-1]["role"] == "assistant"
        assert updated_messages[-1]["role"] == "tool"
        return SimpleNamespace(token_ids=list(runtime_token_ids) + [99])


def sse_events(body: bytes):
    events = []
    for block in body.decode().strip().split("\n\n"):
        if block == "data: [DONE]":
            events.append((None, "[DONE]"))
            continue
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        events.append((event, data))
    return events


class ParserTests(unittest.TestCase):
    def test_qwen_parser_preserves_content_reasoning_and_structured_calls(self) -> None:
        parsed = parse_qwen_assistant(
            "<think>inspect first</think>Working"
            '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>'
        )
        self.assertEqual("inspect first", parsed.reasoning)
        self.assertEqual("Working", parsed.content)
        self.assertEqual("read_file", parsed.tool_calls[0]["name"])
        self.assertEqual('{"path":"README.md"}', parsed.tool_calls[0]["arguments"])

    def test_qwen_parser_rejects_malformed_tool_json(self) -> None:
        with self.assertRaisesRegex(BridgeError, "not JSON"):
            parse_qwen_assistant("<tool_call>{broken}</tool_call>")


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_completion_hides_terminal_eos_without_changing_training_tokens(self) -> None:
        bridge = VerlOpenAIBridge(
            FakeManager(),
            FakeTokenizer("hello<|im_end|>"),
            {},
            model="qwen",
        )
        events = sse_events(
            await bridge.complete(
                {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
            )
        )
        self.assertEqual("hello", events[0][1]["choices"][0]["delta"]["content"])
        self.assertEqual([20, 21], events[1][1]["response_token_ids"])

    async def test_completion_emits_openai_tool_call_and_exact_training_trace(self) -> None:
        tokenizer = FakeTokenizer(
            "<think>inspect first</think>Working"
            '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>'
        )
        manager = FakeManager()
        bridge = VerlOpenAIBridge(manager, tokenizer, {"temperature": 0.7}, model="qwen")
        body = await bridge.complete(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "inspect"}],
                "tools": [{"type": "function", "function": {"name": "read_file"}}],
            }
        )
        events = sse_events(body)
        delta = events[0][1]["choices"][0]["delta"]
        self.assertEqual("Working", delta["content"])
        self.assertEqual("inspect first", delta["reasoning_content"])
        self.assertEqual("read_file", delta["tool_calls"][0]["function"]["name"])
        self.assertEqual("chock.training_trace.v1", events[1][0])
        self.assertEqual([10, 11], events[1][1]["prompt_token_ids"])
        self.assertEqual([20, 21], events[1][1]["response_token_ids"])
        self.assertEqual([-0.25, -0.5], events[1][1]["old_logprobs"])
        self.assertEqual("tool_calls", events[2][1]["choices"][0]["finish_reason"])
        self.assertEqual("[DONE]", events[3][1])
        self.assertTrue(manager.requests[0]["sampling_params"]["logprobs"])

    async def test_http_endpoint_serves_models_and_chat_sse(self) -> None:
        bridge = VerlOpenAIBridge(FakeManager(), FakeTokenizer("hello"), {}, model="qwen")
        base = await bridge.start()
        try:
            port = int(base.split(":")[2].split("/")[0])
            request = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
                + str(len(request)).encode()
                + b"\r\n\r\n"
                + request
            )
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            self.assertIn(b"HTTP/1.1 200 OK", response)
            self.assertIn(b"Content-Type: text/event-stream", response)
            self.assertIn(b"event: chock.training_trace.v1", response)
        finally:
            await bridge.close()

    async def test_multiturn_requests_preserve_continuous_tokens_and_sticky_routing(self) -> None:
        manager = FakeManager()
        bridge = VerlOpenAIBridge(
            manager,
            FakeTokenizer("hello"),
            {},
            model="qwen",
            request_namespace="group-0-rollout-0",
            continuous_token_builder=FakeContinuousBuilder(),
        )
        await bridge.complete({"messages": [{"role": "user", "content": "inspect"}]})
        await bridge.complete(
            {
                "messages": [
                    {"role": "user", "content": "inspect"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "tool", "content": "result", "tool_call_id": "call_1"},
                ]
            }
        )
        self.assertEqual([10, 11], manager.requests[0]["prompt_ids"])
        self.assertEqual([10, 11, 20, 21, 99], manager.requests[1]["prompt_ids"])
        self.assertEqual(manager.requests[0]["request_id"], manager.requests[1]["request_id"])


if __name__ == "__main__":
    unittest.main()
