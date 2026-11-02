from __future__ import annotations

import json
import unittest

from chock_surgeon.server import (
    ServerError,
    _messages_for_qwen,
    _streaming_delta,
    parse_qwen_tool_calls,
)


class ToolCallParserTests(unittest.TestCase):
    def test_plain_text_is_preserved(self) -> None:
        content, calls = parse_qwen_tool_calls("A direct answer.")
        self.assertEqual("A direct answer.", content)
        self.assertEqual([], calls)

    def test_qwen_function_xml_becomes_openai_tool_call(self) -> None:
        content, calls = parse_qwen_tool_calls(
            "Checking.\n<tool_call>\n<function=read_file>\n"
            "<parameter=path>\n\"src/main.zig\"\n</parameter>\n"
            "<parameter=line>\n12\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        self.assertEqual("Checking.", content)
        self.assertEqual("read_file", calls[0]["function"]["name"])
        self.assertEqual(
            {"path": "src/main.zig", "line": 12},
            json.loads(calls[0]["function"]["arguments"]),
        )

    def test_empty_function_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ServerError, "empty function name"):
            parse_qwen_tool_calls("<tool_call><function= >x</function></tool_call>")

    def test_streamed_tool_calls_have_openai_indices(self) -> None:
        delta = _streaming_delta(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_one",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        )
        self.assertEqual(0, delta["tool_calls"][0]["index"])

    def test_openai_argument_json_is_a_mapping_for_qwen_template(self) -> None:
        messages = _messages_for_qwen(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_one",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                }
            ]
        )
        self.assertEqual(
            {"path": "README.md"},
            messages[0]["tool_calls"][0]["function"]["arguments"],
        )


if __name__ == "__main__":
    unittest.main()
