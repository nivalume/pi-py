"""Port of ``test/env-truncate.test.ts``."""

from __future__ import annotations

from pi_durable.truncate import format_size, truncate_head, utf8_byte_length


def byte_length(content: str) -> int:
    """``new TextEncoder().encode(content).length`` of the UTF-16 string ``content`` stands for."""
    return len(content.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace").encode("utf-8"))


def test_reports_utf8_byte_counts_in_truncation_results() -> None:
    content = "aé🙂\nb"
    result = truncate_head(content, max_bytes=100, max_lines=10)

    assert result["truncated"] is False
    assert result["totalBytes"] == byte_length(content)
    assert result["outputBytes"] == byte_length(content)
    assert result["totalBytes"] == 9


def test_counts_utf8_bytes_like_text_encoder_and_truncates_correctly() -> None:
    # Upstream runs this in a runtime without Buffer; Python has one implementation, measured like TextEncoder.
    inputs = ["", "ascii", "é", "中", "🙂", "\ud83d", "\ude42", "a🙂b", "߿ࠀ￿"]
    assert [utf8_byte_length(value) for value in inputs] == [byte_length(value) for value in inputs]
    assert [utf8_byte_length(value) for value in inputs] == [0, 5, 2, 3, 4, 3, 3, 6, 8]
    head = truncate_head("aé🙂\nb", max_bytes=7, max_lines=10)
    assert (head["content"], head["outputBytes"], head["truncatedBy"]) == ("aé🙂", 7, "bytes")


def test_does_not_count_a_trailing_newline_as_an_extra_line() -> None:
    content = "\n".join(["line"] * 3) + "\n"
    head = truncate_head(content, max_bytes=100, max_lines=3)

    assert (head["truncated"], head["totalLines"], head["outputLines"]) == (False, 3, 3)


def test_truncates_head_by_line_limits() -> None:
    result = truncate_head("one\ntwo\nthree\nfour", max_bytes=100, max_lines=2)
    assert result["content"] == "one\ntwo"
    assert result["truncated"] is True
    assert result["truncatedBy"] == "lines"
    assert result["totalLines"] == 4
    assert result["outputLines"] == 2


def test_reports_bytes_when_only_a_trailing_newline_exceeds_limits_at_the_line_cap() -> None:
    result = truncate_head("hello\nworld\n", max_bytes=11, max_lines=2)
    assert result["content"] == "hello\nworld"
    assert result["truncated"] is True
    assert result["truncatedBy"] == "bytes"
    assert result["totalLines"] == 2
    assert result["outputLines"] == 2


def test_truncates_head_on_utf8_byte_limits_without_partial_lines() -> None:
    result = truncate_head("éé\nabc", max_bytes=4, max_lines=10)

    assert result["content"] == "éé"
    assert result["truncated"] is True
    assert result["truncatedBy"] == "bytes"
    assert result["outputBytes"] == 4
    assert result["firstLineExceedsLimit"] is False


def test_reports_head_truncation_when_the_first_line_exceeds_the_byte_limit() -> None:
    result = truncate_head("éé\nabc", max_bytes=3, max_lines=10)

    assert result["content"] == ""
    assert result["truncated"] is True
    assert result["truncatedBy"] == "bytes"
    assert result["firstLineExceedsLimit"] is True


def test_formats_sizes() -> None:
    assert format_size(1023) == "1023B"
    assert format_size(1536) == "1.5KB"
    assert format_size(3 * 1024 * 1024) == "3.0MB"
    # JavaScript's toFixed rounds the exact binary value half up, where Python's format rounds half to even.
    assert format_size(1280) == "1.3KB"
