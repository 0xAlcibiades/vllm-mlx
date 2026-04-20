# SPDX-License-Identifier: Apache-2.0
"""Tests for Gemma 4 tool call parser."""

import json

from vllm_mlx.tool_parsers.gemma4_tool_parser import Gemma4ToolParser


class TestGemma4ToolParserExtract:
    """Test extract_tool_calls on complete model output."""

    def setup_method(self):
        self.parser = Gemma4ToolParser()

    def test_single_tool_call_string_arg(self):
        output = '<|tool_call>call:read_file{path:<|"|>/tmp/foo.py<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["name"] == "read_file"
        args = json.loads(tc["arguments"])
        assert args == {"path": "/tmp/foo.py"}
        assert result.content is None

    def test_single_tool_call_numeric_arg(self):
        output = "<|tool_call>call:search{limit:10,verbose:false}<tool_call|>"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"limit": 10, "verbose": False}

    def test_mixed_types(self):
        output = '<|tool_call>call:search{query:<|"|>hello world<|"|>,limit:10,verbose:false}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"query": "hello world", "limit": 10, "verbose": False}

    def test_nested_object(self):
        output = '<|tool_call>call:configure{settings:{enabled:true,name:<|"|>test<|"|>}}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"settings": {"enabled": True, "name": "test"}}

    def test_array_argument(self):
        output = '<|tool_call>call:tag{items:[<|"|>foo<|"|>,<|"|>bar<|"|>]}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"items": ["foo", "bar"]}

    def test_multiple_tool_calls_in_one_block(self):
        output = (
            "<|tool_call>"
            'call:glob{pattern:<|"|>README*.md<|"|>}'
            'call:glob{pattern:<|"|>CONTRIBUTING.md<|"|>}'
            "<tool_call|>"
        )
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        args0 = json.loads(result.tool_calls[0]["arguments"])
        args1 = json.loads(result.tool_calls[1]["arguments"])
        assert args0 == {"pattern": "README*.md"}
        assert args1 == {"pattern": "CONTRIBUTING.md"}

    def test_content_before_tool_call(self):
        output = 'Let me read that file for you.\n<|tool_call>call:read_file{path:<|"|>/tmp/foo<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert result.content == "Let me read that file for you."
        assert len(result.tool_calls) == 1

    def test_no_tool_calls(self):
        output = "Hello, how can I help you today?"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == output

    def test_empty_tool_call_block(self):
        output = "<|tool_call><tool_call|>"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is False
        assert result.tool_calls == []

    def test_tool_call_id_generated(self):
        output = '<|tool_call>call:read_file{path:<|"|>/tmp/a<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        tc = result.tool_calls[0]
        assert "id" in tc
        assert tc["id"].startswith("call_")

    def test_string_with_special_chars(self):
        output = '<|tool_call>call:write{content:<|"|>line1\\nline2<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["content"] == "line1\\nline2"

    def test_deeply_nested_objects(self):
        output = "<|tool_call>call:update{a:{b:{c:1,d:true}}}<tool_call|>"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"a": {"b": {"c": 1, "d": True}}}

    def test_null_value(self):
        output = "<|tool_call>call:clear{target:null}<tool_call|>"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"target": None}

    def test_unicode_emoji_in_args(self):
        output = '<|tool_call>call:search{query:<|"|>hello world \U0001f30d \u4f60\u597d<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"query": "hello world \U0001f30d \u4f60\u597d"}

    def test_braces_inside_string_value(self):
        output = '<|tool_call>call:run{code:<|"|>if (x) { return y; }<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"code": "if (x) { return y; }"}

    def test_quoted_keys(self):
        output = '<|tool_call>call:read{<|"|>path<|"|>:<|"|>/tmp/foo<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"path": "/tmp/foo"}

    def test_think_tags_stripped(self):
        output = '<think>Let me think about this...</think><|tool_call>call:search{query:<|"|>test<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1

    def test_missing_end_delimiter(self):
        """Unclosed tool call block still parses (server fallback path)."""
        output = '<|tool_call>call:read_file{path:<|"|>/tmp/foo<|"|>}'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"path": "/tmp/foo"}

    def test_string_with_colon(self):
        """String containing colon pattern must not be corrupted by bare-key quoting."""
        output = '<|tool_call>call:connect{url:<|"|>host:8080<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"url": "host:8080"}

    def test_string_with_newline_and_quote(self):
        """Real newline and double quote inside string values are JSON-escaped."""
        output = '<|tool_call>call:write{text:<|"|>line1\nline2 said "hello"<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"text": 'line1\nline2 said "hello"'}


class TestGemma4ToolParserStreaming:
    """Test streaming tool call extraction."""

    def setup_method(self):
        self.parser = Gemma4ToolParser()
        self.parser.reset()

    def test_streaming_no_tool_call(self):
        """Normal text passes through as content."""
        result = self.parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Hello",
            delta_text="Hello",
        )
        assert result == {"content": "Hello"}

    def test_streaming_progressive_emission(self):
        """Inside an open tool call:

        * content before the call passes through,
        * the delta that opens the call + starts the name is suppressed
          (no ``{`` yet, name incomplete),
        * the delta that completes the name + ``{args...`` emits a
          structured tool_calls chunk carrying the name (empty
          ``arguments`` — the OpenAI API convention for "name known,
          args still streaming").
        """
        r1 = self.parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Sure. ",
            delta_text="Sure. ",
        )
        assert r1 == {"content": "Sure. "}

        r2 = self.parser.extract_tool_calls_streaming(
            previous_text="Sure. ",
            current_text="Sure. <|tool_call>call:read",
            delta_text="<|tool_call>call:read",
        )
        # Name isn't complete until the opening brace arrives.
        assert r2 is None

        r3 = self.parser.extract_tool_calls_streaming(
            previous_text="Sure. <|tool_call>call:read",
            current_text='Sure. <|tool_call>call:read_file{path:<|"|>/tmp/foo<|"|>}',
            delta_text='_file{path:<|"|>/tmp/foo<|"|>}',
        )
        # Progressive streaming: name is emitted as soon as it's parseable.
        assert r3 is not None and "tool_calls" in r3
        tc = r3["tool_calls"][0]
        assert tc["index"] == 0
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "read_file"
        assert tc["function"]["arguments"] == ""

    def test_streaming_emits_on_close(self):
        """Emits structured tool_calls when end delimiter arrives."""
        full_text = (
            'Sure. <|tool_call>call:read_file{path:<|"|>/tmp/foo<|"|>}<tool_call|>'
        )
        result = self.parser.extract_tool_calls_streaming(
            previous_text='Sure. <|tool_call>call:read_file{path:<|"|>/tmp/foo<|"|>}',
            current_text=full_text,
            delta_text="<tool_call|>",
        )
        assert result is not None
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["function"]["name"] == "read_file"
        assert tc["type"] == "function"
        assert tc["index"] == 0


class TestGemma4ToolParserStreamingRegressions:
    """Tests for the bugs fixed in the upstream-port of the streaming
    parser. These all exercise code paths the old vllm-mlx implementation
    silently broke: content after close, partial-prefix leakage, and
    progressive argument streaming.
    """

    def _run_stream(self, tokens):
        """Helper: feed token list through the parser and collect results."""
        parser = Gemma4ToolParser()
        parser.reset()
        accumulated = ""
        events: list[dict] = []
        for tok in tokens:
            prev = accumulated
            accumulated += tok
            r = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=accumulated,
                delta_text=tok,
            )
            if r is not None:
                events.append(r)
        return events

    def test_content_after_closed_tool_call_not_dropped(self):
        """After <tool_call|> arrives, subsequent plain text must still
        reach the client as content. The old vllm-mlx parser fell through
        to ``return None`` once ``has_start=True`` was latched, silently
        dropping everything after the first close.
        """
        tokens = [
            "I'll read it. ",
            "<|tool_call>",
            "call:read_file",
            '{path:<|"|>/tmp/x<|"|>}',
            "<tool_call|>",
            "All done.",
        ]
        events = self._run_stream(tokens)

        contents = [e["content"] for e in events if "content" in e]
        tool_events = [e for e in events if "tool_calls" in e]

        # Prior content before the call and post-call content both appear.
        full_content = "".join(contents)
        assert "I'll read it." in full_content
        assert "All done." in full_content
        # Neither delimiter leaks into content.
        assert "<|tool_call>" not in full_content
        assert "<tool_call|>" not in full_content
        # The tool call itself was emitted (at least one structured chunk).
        assert tool_events, "tool call should have produced structured events"

    def test_partial_start_token_at_delta_boundary_is_buffered(self):
        """A delta that is itself a proper prefix of ``<|tool_call>`` must
        be buffered — it must NOT reach the client as literal content.
        Once the next delta completes the start token, the parser
        transitions into tool-call mode.
        """
        parser = Gemma4ToolParser()
        parser.reset()

        # Delta #1: buffered entirely (proper prefix of start token,
        # nothing else to emit). Returning None is the correct outcome.
        curr1 = "<|tool"
        r1 = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=curr1,
            delta_text="<|tool",
        )
        assert r1 is None or "<|tool" not in (r1.get("content") or "")

        # Delta #2: completes the start token.
        prev2, curr2 = curr1, curr1 + "_call>"
        r2 = parser.extract_tool_calls_streaming(
            previous_text=prev2,
            current_text=curr2,
            delta_text="_call>",
        )
        # State has advanced into a tool call.
        assert "<|tool_call>" in curr2
        assert parser.current_tool_id == 0
        # Neither delta leaked the partial prefix as content.
        emitted = [r1, r2]
        content_so_far = "".join(
            (r.get("content") or "") for r in emitted if isinstance(r, dict)
        )
        assert "<|tool" not in content_so_far

    def test_progressive_streaming_emits_name_then_args(self):
        """Within a tool call, the name is emitted first (empty args),
        then subsequent argument content is diffed incrementally. The
        final close flushes any remaining suffix so the concatenated
        ``arguments`` stream is a valid JSON object.
        """
        tokens = [
            "<|tool_call>",
            "call:search",
            "{query:",
            '<|"|>hello world<|"|>',
            ",limit:10",
            "}",
            "<tool_call|>",
        ]
        events = self._run_stream(tokens)

        # All structured events for tool index 0.
        tc_events = [
            tc
            for e in events
            if "tool_calls" in e
            for tc in e["tool_calls"]
            if tc.get("index") == 0
        ]
        assert tc_events, "expected at least one structured tool_call event"

        # First structured chunk carries the name.
        first = tc_events[0]
        assert first["function"]["name"] == "search"
        assert first["function"]["arguments"] == ""
        assert "id" in first and first["id"].startswith("call_")

        # Concatenate argument fragments — must parse as valid JSON.
        args_stream = "".join(
            tc["function"].get("arguments", "") for tc in tc_events
        )
        parsed = json.loads(args_stream)
        assert parsed == {"query": "hello world", "limit": 10}


class TestGemma4Registration:
    """Test parser registration and flags."""

    def test_registered_in_manager(self):
        from vllm_mlx.tool_parsers import ToolParserManager

        parser_cls = ToolParserManager.get_tool_parser("gemma4")
        assert parser_cls is Gemma4ToolParser

    def test_native_format_false(self):
        assert Gemma4ToolParser.SUPPORTS_NATIVE_TOOL_FORMAT is False
        assert Gemma4ToolParser.supports_native_format() is False
