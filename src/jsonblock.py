"""Shared JSON extraction for model output in the climate-arXiv RAG project (src/).

Every time the pipeline requests JSON from a model (for question generation, reranking, judging),
it receives output wrapped in prose. Each parser extracts the content from the first opening
bracket to the last closing bracket and attempts json.loads. This logic, which was previously
duplicated four times, is now centralized in a single version. It returns None for any failure such as a missing
bracket, malformed JSON, or incorrect root type—allowing callers to distinguish between a parse
failure and a valid result instead of silently defaulting.

"""

from __future__ import annotations

import json


def span(text, start):
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    quoted = False
    escaped = False
    for i in range(start, len(text)):
        c = text[i]
        if quoted:
            escaped = c == "\\" and not escaped
            if c == '"' and not escaped:
                quoted = False
        elif c == '"':
            quoted = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def json_block(text):
    if not text:
        return None
    found = None
    i = 0
    while i < len(text):
        if text[i] in "{[":
            end = span(text, i)
            if end:
                try:
                    value = json.loads(text[i:end])
                    if value:
                        found = value
                    i = end
                    continue
                except json.JSONDecodeError:
                    pass
        i += 1
    return found