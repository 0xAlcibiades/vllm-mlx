# SPDX-License-Identifier: Apache-2.0
"""
Tool call parser for Google Gemma 4 models — ported from upstream vLLM.

Ref: vllm/tool_parsers/gemma4_tool_parser.py  (streaming parser)
Ref: vllm/tool_parsers/gemma4_utils.py        (offline helpers)

Gemma 4's tool call format::

    <|tool_call>call:func_name{key:<|"|>value<|"|>,count:42}<tool_call|>

- ``<|tool_call>`` / ``<tool_call|>`` delimit tool call blocks
- ``<|"|>`` replaces ``"`` for string values
- Keys are bare identifiers (no quotes)
- Multiple ``call:name{...}`` may appear in a single block
- Function names may contain letters, digits, underscores, hyphens,
  and dots (e.g. ``get-weather``, ``module.func``)

Streaming strategy is accumulate-then-parse-then-diff (matches upstream):

1. Accumulate the raw Gemma 4 argument string as deltas arrive.
2. Parse it with ``_parse_gemma4_args()`` into a Python dict.
3. Serialize to JSON with ``json.dumps()``.
4. Withhold trailing closing chars (``}``, ``"``, ``]``, partial
   ``<|"|>`` fragments) to avoid corruption as more tokens arrive.
5. Diff against the previously-streamed JSON and emit only the new
   fragment.

Multi-token special sequences are buffered via ``_buffer_delta_text``
so a partial ``<|tool`` fragment never leaks into content. Content
emitted after a fully-closed tool call correctly passes through as
content (the previous implementation silently dropped it).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

logger = logging.getLogger(__name__)

# Delimiters (also token IDs 48/49 in Gemma 4, but we parse at the string level)
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'


# ---------------------------------------------------------------------------
# Argument parser — matches upstream vLLM's gemma4_tool_parser
# ---------------------------------------------------------------------------


def _parse_gemma4_value(value_str: str) -> object:
    """Parse a single Gemma 4 value (right of ``key:``) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str

    if value_str == "true":
        return True
    if value_str == "false":
        return False
    if value_str.lower() in ("null", "none", "nil"):
        return None

    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    return value_str


def _parse_gemma4_args(args_str: str, *, partial: bool = False) -> dict:
    """Parse Gemma 4's ``key:value`` format into a Python dict.

    Supported shapes::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Args:
        args_str: Raw Gemma 4 argument text (no surrounding ``{``/``}``).
        partial: When True (streaming), bare values at end of string are
            withheld because they may be incomplete and type-unstable
            (e.g. a partial boolean parsed as a bare string).

    Returns:
        A dict ready for ``json.dumps()``.
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1

        if i >= n:
            if not partial:
                result[key] = ""
            break

        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break

        # String value: <|"|>...<|"|>
        if args_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)

        # Nested object
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = args_str.find(STRING_DELIM, i)
                    i = n if nd == -1 else nd + len(STRING_DELIM)
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        # Array
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = args_str.find(STRING_DELIM, i)
                    i = n if nd == -1 else nd + len(STRING_DELIM)
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])

        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list:
    """Parse a Gemma 4 array body into a Python list."""
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)

        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_args(arr_str[obj_start:i], partial=True))
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_array(arr_str[sub_start:i], partial=True))
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


# ---------------------------------------------------------------------------
# Helpers inlined from vllm.* (upstream parser imports these; vllm-mlx
# doesn't depend on the vllm API, so we reimplement locally).
# ---------------------------------------------------------------------------


def _find_common_prefix(a: str, b: str) -> str:
    """Longest string that is a prefix of both ``a`` and ``b``."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def _make_tool_call_id() -> str:
    """OpenAI-style tool call identifier."""
    return f"call_{uuid.uuid4().hex[:24]}"


# Regex matching a complete tool call — used by the streaming parser's
# end-of-call flush. Function names may contain letters, digits,
# underscores, hyphens, and dots.
_TOOL_CALL_REGEX = re.compile(
    r"<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)

# Helpers kept from the pre-port non-streaming parser. These handle
# edge cases the streaming-oriented _parse_gemma4_args cannot (multiple
# ``call:name{...}`` inside a single ``<|tool_call>...<tool_call|>``
# block, and ``<|"|>``-quoted keys).

_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")
_STRING_DELIM_RE = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)
_CALL_PREFIX = re.compile(r"call:([\w\-\.]+)\s*\{")
_BARE_KEY = re.compile(r"(?<=[{,])\s*([\w\-\.]+)\s*:")
_MAX_ARG_BLOCK_LEN = 1_048_576


def _find_balanced_brace(text: str, start: int) -> int:
    """Find the index of the closing ``}`` balancing the ``{`` at ``start``.

    Skips over ``<|"|>``-delimited string regions so braces inside string
    values don't affect depth counting.
    """
    if len(text) - start > _MAX_ARG_BLOCK_LEN:
        return -1
    depth = 0
    i = start
    in_string = False
    n = len(text)
    while i < n:
        if text.startswith(STRING_DELIM, i):
            in_string = not in_string
            i += len(STRING_DELIM)
            continue
        if not in_string:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _gemma4_args_to_json(text: str) -> str:
    """Convert a Gemma 4 ``{...}``-delimited argument body into valid JSON.

    Three-step conversion (ORDER MATTERS):

    1. Extract ``<|"|>``-delimited strings into numbered ``\\x00N\\x00``
       placeholders — protects string contents from the bare-key quoting
       in step 2.
    2. Quote bare keys (``word:`` → ``"word":``).
    3. Restore placeholders as properly JSON-escaped strings.
    """
    strings: list[str] = []

    def _capture(m: re.Match) -> str:
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    text = _STRING_DELIM_RE.sub(_capture, text)
    text = _BARE_KEY.sub(r'"\1":', text)

    def _restore(m: re.Match) -> str:
        idx = int(m.group(1))
        return json.dumps(strings[idx]) if idx < len(strings) else m.group(0)

    text = _PLACEHOLDER_RE.sub(_restore, text)
    return text


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@ToolParserManager.register_module("gemma4")
class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Google Gemma 4 models.

    **Non-streaming path**: regex match on the complete output, then parse
    each ``call:name{args}`` body with ``_parse_gemma4_args``.

    **Streaming path**: tag-counting state machine.

    1. Each delta runs through ``_buffer_delta_text`` so partial
       ``<|tool_call>`` / ``<tool_call|>`` fragments never leak as content.
    2. Before any tool call starts, deltas pass through as content.
    3. A new ``<|tool_call>`` opens a tool call. The function name is
       emitted in the first structured chunk.
    4. While inside the call, arguments are parsed partially,
       JSON-serialized, and diffed against the previously-emitted JSON.
       Trailing closing chars that might shift are withheld.
    5. A ``<tool_call|>`` closes the call; a final flush emits any
       remaining argument suffix.
    6. After a completed tool call, subsequent deltas once again pass
       through as content — fixes the "content dropped after first tool
       call" bug in the previous vllm-mlx implementation.

    Return shape matches vllm-mlx's dict convention:

    * ``{"content": str}`` for plain text deltas
    * ``{"tool_calls": [{"index": int, "id"?: str, "type"?: "function",
      "function": {"name"?: str, "arguments"?: str}}]}`` for structured
      tool-call chunks
    * ``None`` to suppress the delta (buffered or no-op)
    """

    def __init__(
        self,
        tokenizer: Any = None,
        tools: list[dict[str, Any]] | None = None,
    ):
        super().__init__(tokenizer)
        self.tool_call_start_token = TOOL_CALL_START
        self.tool_call_end_token = TOOL_CALL_END
        self.tool_call_regex = _TOOL_CALL_REGEX
        self._reset_streaming_state()
        self.buffered_delta_text: str = ""

    # ---- streaming state ------------------------------------------------

    def _reset_streaming_state(self) -> None:
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

    def reset(self) -> None:
        """Reset parser state for a new request."""
        self._reset_streaming_state()
        self.buffered_delta_text = ""

    # ---- delta buffering for multi-token special sequences --------------

    def _buffer_delta_text(self, delta_text: str) -> str:
        """Hold back partial prefixes of ``<|tool_call>`` / ``<tool_call|>``
        so they never leak into content. Returns the safe-to-emit portion.
        """
        combined = self.buffered_delta_text + delta_text

        # Complete tag — release.
        if combined.endswith(TOOL_CALL_START) or combined.endswith(TOOL_CALL_END):
            self.buffered_delta_text = ""
            return combined

        # Partial prefix of either tag — withhold that trailing suffix.
        for tag in (TOOL_CALL_START, TOOL_CALL_END):
            for i in range(len(tag) - 1, 0, -1):
                if combined.endswith(tag[:i]):
                    self.buffered_delta_text = combined[-i:]
                    return combined[:-i]

        self.buffered_delta_text = ""
        return combined

    # ---- non-streaming --------------------------------------------------

    def extract_tool_calls(
        self,
        model_output: str,
        request: dict[str, Any] | None = None,
    ) -> ExtractedToolCallInformation:
        """Extract all tool calls from a complete model response.

        Block-scans between ``<|tool_call>`` / ``<tool_call|>`` markers,
        then within each block matches each ``call:name{args}`` using
        balanced-brace scanning. This correctly handles:

        * Multiple ``call:name{...}`` inside a single ``<|tool_call>`` /
          ``<tool_call|>`` block.
        * ``<|"|>``-quoted keys (placeholder substitution via
          ``_gemma4_args_to_json``).
        * Braces inside string values (balanced-brace scan skips over
          ``<|"|>``-delimited regions).
        """
        cleaned = self.strip_think_tags(model_output)

        start_idx = cleaned.find(self.tool_call_start_token)
        if start_idx == -1:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        content_before = cleaned[:start_idx].strip() or None

        # Collect each <|tool_call>...<tool_call|> block's interior text.
        blocks: list[str] = []
        pos = 0
        while True:
            b_start = cleaned.find(self.tool_call_start_token, pos)
            if b_start == -1:
                break
            body_start = b_start + len(self.tool_call_start_token)
            b_end = cleaned.find(self.tool_call_end_token, body_start)
            if b_end == -1:
                blocks.append(cleaned[body_start:])
                break
            blocks.append(cleaned[body_start:b_end])
            pos = b_end + len(self.tool_call_end_token)

        tool_calls: list[dict[str, Any]] = []
        for block in blocks:
            bpos = 0
            while bpos < len(block):
                m = _CALL_PREFIX.search(block, bpos)
                if not m:
                    break
                func_name = m.group(1)
                brace_start = m.end() - 1
                brace_end = _find_balanced_brace(block, brace_start)
                if brace_end == -1:
                    bpos = m.end()
                    continue
                args_raw = block[brace_start : brace_end + 1]
                try:
                    args_json = _gemma4_args_to_json(args_raw)
                    json.loads(args_json)  # validate
                    tool_calls.append(
                        {
                            "id": _make_tool_call_id(),
                            "name": func_name,
                            "arguments": args_json,
                        }
                    )
                except (json.JSONDecodeError, ValueError):
                    logger.warning(
                        "Gemma 4 tool parser: failed to parse args for call:%s",
                        func_name,
                    )
                bpos = brace_end + 1

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content_before,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    # ---- streaming ------------------------------------------------------

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Streaming entrypoint. Returns a vllm-mlx-shaped dict, or None."""
        safe_delta = self._buffer_delta_text(delta_text)

        # Fast path: no tool call has started in the accumulated stream yet.
        if self.tool_call_start_token not in current_text:
            if safe_delta:
                return {"content": safe_delta}
            return None

        try:
            return self._extract_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=safe_delta,
            )
        except Exception:
            logger.exception("Gemma 4 streaming tool call extraction failed")
            return None

    def _extract_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> dict[str, Any] | None:
        """Tag-counting state machine over current_text vs previous_text."""
        start_tok = self.tool_call_start_token
        end_tok = self.tool_call_end_token

        start_count = current_text.count(start_tok)
        end_count = current_text.count(end_tok)
        prev_start_count = previous_text.count(start_tok)
        prev_end_count = previous_text.count(end_tok)

        # Case 1: all open tool calls are closed and nothing new happened
        # in this delta — emit as content. (Fixes the "content dropped
        # after first tool call" bug.)
        if (
            start_count == end_count
            and prev_end_count == end_count
            and end_tok not in delta_text
        ):
            if delta_text:
                clean = delta_text.replace(start_tok, "").replace(end_tok, "")
                if clean:
                    return {"content": clean}
            return None

        # Case 2: a new tool call just opened in this delta.
        if start_count > prev_start_count and start_count > end_count:
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool.append("")
            self.prev_tool_call_arr.append({})
            if len(delta_text) <= len(start_tok):
                return None
            # else fall through — delta may contain more than the token itself

        # Case 3: a tool call just closed in this delta.
        if end_count > prev_end_count:
            return self._handle_tool_call_end(current_text)

        # Case 4: inside an active tool call.
        if start_count > end_count:
            return self._handle_tool_call_middle(current_text)

        # Default: content outside calls — scrub any stray tag remnants.
        if delta_text:
            clean = delta_text.replace(start_tok, "").replace(end_tok, "")
            if clean:
                return {"content": clean}
        return None

    # ---- phase handlers -------------------------------------------------

    def _extract_partial_call(
        self, current_text: str
    ) -> tuple[str | None, str]:
        """Parse function name + raw argument body out of the in-flight
        ``<|tool_call>call:name{args...`` region.

        Returns ``(None, "")`` if the prefix isn't fully parseable yet.
        """
        last_start = current_text.rfind(self.tool_call_start_token)
        if last_start == -1:
            return None, ""

        partial_call = current_text[
            last_start + len(self.tool_call_start_token) :
        ]

        if self.tool_call_end_token in partial_call:
            partial_call = partial_call.split(self.tool_call_end_token)[0]

        if not partial_call.startswith("call:"):
            return None, ""

        func_part = partial_call[5:]
        if "{" not in func_part:
            return None, ""

        func_name, _, args_part = func_part.partition("{")
        func_name = func_name.strip()

        if args_part.endswith("}"):
            args_part = args_part[:-1]

        return func_name, args_part

    def _handle_tool_call_middle(
        self, current_text: str
    ) -> dict[str, Any] | None:
        """Inside an active tool call: emit name once, then arg diffs."""
        func_name, args_part = self._extract_partial_call(current_text)
        if func_name is None:
            return None

        if not self.current_tool_name_sent and func_name:
            self.current_tool_name_sent = True
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": func_name,
                "arguments": {},
            }
            return {
                "tool_calls": [
                    {
                        "index": self.current_tool_id,
                        "type": "function",
                        "id": _make_tool_call_id(),
                        "function": {
                            "name": func_name,
                            "arguments": "",
                        },
                    }
                ]
            }

        if self.current_tool_name_sent and args_part:
            return self._emit_argument_diff(args_part)

        return None

    def _handle_tool_call_end(
        self, current_text: str
    ) -> dict[str, Any] | None:
        """A closing marker just arrived — flush the final argument diff.

        If per-call state was never built up progressively (e.g. the
        parser was instantiated mid-stream, or a client chose to call
        streaming entrypoints only on the close delta), fall back to
        emitting all completed tool calls from the accumulated text in
        one structured batch. This preserves the "fresh parser, final
        delta" contract the older vllm-mlx implementation had.
        """
        all_matches = self.tool_call_regex.findall(current_text)
        if not all_matches:
            return None

        if self.current_tool_id < 0 or self.current_tool_id >= len(
            self.prev_tool_call_arr
        ):
            # No progressive state — emit every completed call as a
            # full tool_calls batch (name + args in one shot).
            return {
                "tool_calls": [
                    {
                        "index": i,
                        "id": _make_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                _parse_gemma4_args(args), ensure_ascii=False
                            ),
                        },
                    }
                    for i, (name, args) in enumerate(all_matches)
                ]
            }

        if self.current_tool_id >= len(all_matches):
            return None

        _, args_str = all_matches[self.current_tool_id]
        final_args = _parse_gemma4_args(args_str)
        final_args_json = json.dumps(final_args, ensure_ascii=False)

        prev_streamed = self.streamed_args_for_tool[self.current_tool_id]
        if len(final_args_json) <= len(prev_streamed):
            return None

        diff = final_args_json[len(prev_streamed) :]
        self.streamed_args_for_tool[self.current_tool_id] = final_args_json
        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = final_args

        return {
            "tool_calls": [
                {
                    "index": self.current_tool_id,
                    "function": {"arguments": diff},
                }
            ]
        }

    def _emit_argument_diff(
        self, raw_args_str: str
    ) -> dict[str, Any] | None:
        """Accumulate-parse-diff for the in-flight argument body.

        The parse is partial; the produced JSON is trimmed of trailing
        closing chars that may shift next delta so the client never sees
        a ``}`` that later turns into ``, "more": …}``.
        """
        try:
            current_args = _parse_gemma4_args(raw_args_str, partial=True)
        except Exception:
            return None

        if not current_args:
            return None

        current_args_json = json.dumps(current_args, ensure_ascii=False)

        safe_json = current_args_json
        while safe_json and safe_json[-1] in ("}", '"', "]", "<", "|", "\\", ">"):
            safe_json = safe_json[:-1]

        prev_streamed = self.streamed_args_for_tool[self.current_tool_id]
        if not safe_json or safe_json == prev_streamed:
            return None

        if prev_streamed:
            prefix = _find_common_prefix(prev_streamed, safe_json)
            if len(prefix) < len(prev_streamed):
                # Structure shifted — retract our tracking to the shared
                # prefix and wait for _handle_tool_call_end to flush.
                self.streamed_args_for_tool[self.current_tool_id] = prefix
                return None
            diff = safe_json[len(prev_streamed) :]
        else:
            diff = safe_json

        if not diff:
            return None

        self.streamed_args_for_tool[self.current_tool_id] = safe_json
        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = current_args

        return {
            "tool_calls": [
                {
                    "index": self.current_tool_id,
                    "function": {"arguments": diff},
                }
            ]
        }
