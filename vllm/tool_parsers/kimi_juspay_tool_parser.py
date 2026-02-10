# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Kimi Juspay Tool Call Parser with GLM4-style incremental streaming.

This parser uses GLM4's buffer consumption approach to stream tool calls
incrementally without buffer size limits. Content before tool calls is
returned immediately, providing a true streaming experience.

Key differences from kimi_k2_tool_parser.py:
- No buffer size limits (incremental consumption instead)
- Content before tool section returned immediately
- String-based marker detection instead of token ID counting
- State machine approach with boolean flags
"""

import json
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)


class KimiJuspayToolParser(ToolParser):
    """Tool parser for Kimi K2 models with GLM4-style incremental streaming.

    This parser emits tool-call deltas incrementally as arguments arrive.
    Content before tool sections is returned immediately to the user,
    eliminating the need for buffer size limits.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # Kimi K2 markers
        self.tool_calls_start_token = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token = "<|tool_calls_section_end|>"
        self.tool_calls_start_token_variants = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_section_begin|>",  # singular variant
        ]
        self.tool_calls_end_token_variants = [
            "<|tool_calls_section_end|>",
            "<|tool_call_section_end|>",  # singular variant
        ]
        self.tool_call_start_token = "<|tool_call_begin|>"
        self.tool_call_end_token = "<|tool_call_end|>"
        self.tool_call_arg_begin = "<|tool_call_argument_begin|>"

        # Regex for non-streaming extraction (same as kimi_k2)
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*"
            r"<\|tool_call_argument_begin\|>\s*"
            r"(?P<function_arguments>(?:(?!<\|tool_call_begin\|>).)*?)\s*"
            r"<\|tool_call_end\|>",
            re.DOTALL,
        )

        # Streaming state (GLM4-style)
        self._buffer: str = ""
        self._in_tool_section: bool = False
        self._in_tool_call: bool = False
        self._current_tool_name: str | None = None
        self._current_tool_id_str: str | None = None
        self._streaming_args: bool = False
        self._tool_name_sent: bool = False

        # Tool call tracking
        self.current_tool_id: int = -1
        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.streamed_args_for_tool: list[str] = []
        self._tool_call_ids: list[str] = []

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

    def reset_streaming_state(self) -> None:
        """Reset all streaming state between requests."""
        self._buffer = ""
        self._in_tool_section = False
        self._in_tool_call = False
        self._current_tool_name = None
        self._current_tool_id_str = None
        self._streaming_args = False
        self._tool_name_sent = False
        self.current_tool_id = -1
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self._tool_call_ids = []
        logger.debug("Streaming state reset")

    def _find_section_start(self, text: str) -> int:
        """Find the start of any tool section marker variant."""
        positions = []
        for variant in self.tool_calls_start_token_variants:
            pos = text.find(variant)
            if pos != -1:
                positions.append(pos)
        return min(positions) if positions else -1

    def _find_section_end(self, text: str) -> int:
        """Find the end of any tool section marker variant."""
        positions = []
        for variant in self.tool_calls_end_token_variants:
            pos = text.find(variant)
            if pos != -1:
                positions.append(pos)
        return min(positions) if positions else -1

    def _get_section_start_len(self, text: str) -> int:
        """Get the length of the section start marker found at position 0."""
        for variant in self.tool_calls_start_token_variants:
            if text.startswith(variant):
                return len(variant)
        return len(self.tool_calls_start_token)

    def _get_section_end_len(self, text: str, pos: int) -> int:
        """Get the length of the section end marker at given position."""
        for variant in self.tool_calls_end_token_variants:
            if text[pos:].startswith(variant):
                return len(variant)
        return len(self.tool_calls_end_token)

    def _safe_flush_buffer(self) -> str:
        """Safely flush buffer, checking for partial markers at end."""
        # Check for partial section start marker at end
        for i in range(1, len(self.tool_calls_start_token)):
            for variant in self.tool_calls_start_token_variants:
                if self._buffer.endswith(variant[:i]):
                    out = self._buffer[:-i]
                    self._buffer = self._buffer[-i:]
                    return out
        # No partial marker, flush all
        out = self._buffer
        self._buffer = ""
        return out

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (non-streaming)."""
        # Check for section start marker
        section_start_pos = self._find_section_start(model_output)
        if section_start_pos == -1:
            # Check if we have end markers without start (malformed)
            if self._find_section_end(model_output) != -1:
                logger.warning(
                    "MALFORMED TOOL CALL: Found section_end but not section_begin. "
                    "First 500 chars: %s",
                    model_output[:500],
                )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            # Extract tool calls using regex
            function_call_tuples = self.tool_call_regex.findall(model_output)
            logger.debug("function_call_tuples: %s", function_call_tuples)

            tool_calls = []
            for match in function_call_tuples:
                function_id, function_args = match
                # function_id: functions.get_weather:0 or get_weather:0
                function_name = function_id.split(":")[0].split(".")[-1]
                tool_calls.append(
                    ToolCall(
                        id=function_id,
                        type="function",
                        function=FunctionCall(
                            name=function_name, arguments=function_args
                        ),
                    )
                )

            content = model_output[:section_start_pos]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Extract tool calls incrementally using GLM4-style buffer consumption."""
        self._buffer += delta_text

        while True:
            # State: Not in tool section yet
            if not self._in_tool_section:
                start_idx = self._find_section_start(self._buffer)

                if start_idx == -1:
                    # No section start found - return content immediately
                    out = self._safe_flush_buffer()
                    return DeltaMessage(content=out) if out else None

                if start_idx > 0:
                    # Content before tool section - return it immediately
                    out = self._buffer[:start_idx]
                    self._buffer = self._buffer[start_idx:]
                    return DeltaMessage(content=out)

                # Enter tool section - consume the marker
                marker_len = self._get_section_start_len(self._buffer)
                self._buffer = self._buffer[marker_len:]
                self._in_tool_section = True
                logger.debug("Entering tool section")
                continue

            # State: In tool section, check for section end
            section_end_idx = self._find_section_end(self._buffer)
            if section_end_idx != -1 and not self._in_tool_call:
                # Section is ending and we're not in a tool call
                marker_len = self._get_section_end_len(self._buffer, section_end_idx)
                # Content after section end
                post_content = self._buffer[section_end_idx + marker_len:]
                self._buffer = post_content
                self._in_tool_section = False
                logger.debug("Exiting tool section")
                # Return any content after the section
                if post_content.strip():
                    out = self._safe_flush_buffer()
                    return DeltaMessage(content=out) if out else None
                continue

            # State: In tool section, not in a tool call
            if not self._in_tool_call:
                tool_start_idx = self._buffer.find(self.tool_call_start_token)

                if tool_start_idx == -1:
                    # Check for section end (empty section or between tools)
                    section_end_idx = self._find_section_end(self._buffer)
                    if section_end_idx != -1:
                        marker_len = self._get_section_end_len(
                            self._buffer, section_end_idx
                        )
                        post_content = self._buffer[section_end_idx + marker_len:]
                        self._buffer = post_content
                        self._in_tool_section = False
                        logger.debug("Exiting tool section (no more tools)")
                        if post_content.strip():
                            out = self._safe_flush_buffer()
                            return DeltaMessage(content=out) if out else None
                        continue
                    # Wait for more data
                    return None

                if tool_start_idx > 0:
                    # Noise/whitespace before tool_call_begin - skip it
                    self._buffer = self._buffer[tool_start_idx:]
                    continue

                # Start new tool call
                self._buffer = self._buffer[len(self.tool_call_start_token):]
                self._in_tool_call = True
                self._tool_name_sent = False
                self._streaming_args = False
                self.current_tool_id += 1
                self._ensure_tool_state()
                logger.debug("Starting tool call %d", self.current_tool_id)
                continue

            # State: In a tool call, need to parse tool ID and args
            if not self._tool_name_sent:
                # Looking for tool ID (e.g., "functions.get_weather:0")
                arg_begin_idx = self._buffer.find(self.tool_call_arg_begin)
                tool_end_idx = self._buffer.find(self.tool_call_end_token)

                # Determine delimiter
                if arg_begin_idx == -1 and tool_end_idx == -1:
                    # Wait for more data
                    return None

                # Choose the first delimiter
                if arg_begin_idx != -1 and (
                    tool_end_idx == -1 or arg_begin_idx < tool_end_idx
                ):
                    delimiter_idx = arg_begin_idx
                    is_arg_begin = True
                else:
                    delimiter_idx = tool_end_idx
                    is_arg_begin = False

                # Extract tool ID
                tool_id_str = self._buffer[:delimiter_idx].strip()
                if not tool_id_str:
                    # Empty tool call - skip
                    if is_arg_begin:
                        self._buffer = self._buffer[
                            delimiter_idx + len(self.tool_call_arg_begin):
                        ]
                    else:
                        self._buffer = self._buffer[
                            delimiter_idx + len(self.tool_call_end_token):
                        ]
                        self._in_tool_call = False
                    continue

                # Parse tool name from ID (e.g., "functions.get_weather:0" -> "get_weather")
                tool_name = tool_id_str.split(":")[0].split(".")[-1]
                self._current_tool_name = tool_name
                self._current_tool_id_str = tool_id_str
                self._tool_name_sent = True

                if is_arg_begin:
                    self._buffer = self._buffer[
                        delimiter_idx + len(self.tool_call_arg_begin):
                    ]
                    self._streaming_args = True
                else:
                    # No arguments
                    self._buffer = self._buffer[
                        delimiter_idx + len(self.tool_call_end_token):
                    ]
                    self._in_tool_call = False
                    self.streamed_args_for_tool[self.current_tool_id] = "{}"
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": tool_name,
                        "arguments": {},
                    }

                # Emit tool name delta
                return self._emit_tool_name_delta(tool_id_str, tool_name)

            # State: Streaming arguments
            if self._streaming_args:
                tool_end_idx = self._buffer.find(self.tool_call_end_token)

                if tool_end_idx == -1:
                    # Still streaming args - check for partial marker
                    safe_len = len(self._buffer)
                    for i in range(1, len(self.tool_call_end_token)):
                        if self._buffer.endswith(self.tool_call_end_token[:i]):
                            safe_len = len(self._buffer) - i
                            break

                    if safe_len > 0:
                        to_emit = self._buffer[:safe_len]
                        self._buffer = self._buffer[safe_len:]
                        self.streamed_args_for_tool[self.current_tool_id] += to_emit
                        return self._emit_tool_args_delta(to_emit)
                    return None

                # Found end of tool call
                args_content = self._buffer[:tool_end_idx].strip()
                self._buffer = self._buffer[
                    tool_end_idx + len(self.tool_call_end_token):
                ]
                self._in_tool_call = False
                self._streaming_args = False

                # Store final args
                full_args = self.streamed_args_for_tool[self.current_tool_id] + args_content
                self.streamed_args_for_tool[self.current_tool_id] = full_args

                # Try to parse args
                try:
                    args_dict = json.loads(full_args)
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": self._current_tool_name,
                        "arguments": args_dict,
                    }
                except json.JSONDecodeError:
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": self._current_tool_name,
                        "arguments": full_args,
                    }

                if args_content:
                    return self._emit_tool_args_delta(args_content)
                continue

            # Shouldn't reach here
            return None

    def _ensure_tool_state(self) -> None:
        """Ensure tool state arrays are properly sized."""
        while len(self._tool_call_ids) <= self.current_tool_id:
            self._tool_call_ids.append(f"call_{self.current_tool_id}")
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})

    def _emit_tool_name_delta(self, tool_id: str, tool_name: str) -> DeltaMessage:
        """Emit a delta message with the tool name."""
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    id=tool_id,
                    type="function",
                    function=DeltaFunctionCall(
                        name=tool_name,
                        arguments="",
                    ).model_dump(exclude_none=True),
                )
            ]
        )

    def _emit_tool_args_delta(self, fragment: str) -> DeltaMessage:
        """Emit a delta message with argument fragment."""
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    function=DeltaFunctionCall(arguments=fragment).model_dump(
                        exclude_none=True
                    ),
                )
            ]
        )
