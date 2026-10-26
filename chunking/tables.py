"""Helpers for keeping table rows intact while chunking."""

from __future__ import annotations

from chunking.tokens import count_tokens


def split_into_blocks(text: str) -> list[tuple[str, str]]:
    """Split text into alternating text and markdown-table blocks."""

    blocks: list[tuple[str, str]] = []
    current: list[str] = []
    current_type: str | None = None

    for line in text.splitlines():
        line_type = "table" if line.lstrip().startswith("|") else "text"
        if current_type is None:
            current_type = line_type
            current = [line]
        elif line_type == current_type:
            current.append(line)
        else:
            blocks.append((current_type, "\n".join(current)))
            current_type = line_type
            current = [line]

    if current and current_type is not None:
        blocks.append((current_type, "\n".join(current)))

    return blocks


def split_table_block(block: str, budget_tokens: int) -> list[str]:
    """Split a markdown table into row-preserving chunks."""

    lines = [line for line in block.splitlines() if line.strip()]
    if not lines:
        return []

    header_lines: list[str] = []
    rows: list[str] = lines
    if len(lines) >= 2 and lines[1].lstrip().startswith("|"):
        separator = lines[1].replace("|", "").replace("-", "").replace(":", "").strip()
        if not separator:
            header_lines = lines[:2]
            rows = lines[2:]

    chunks: list[str] = []
    current = list(header_lines)

    for row in rows:
        candidate = [*current, row]
        if count_tokens("\n".join(candidate)) <= budget_tokens or len(current) == len(header_lines):
            current = candidate
            continue

        if len(current) > len(header_lines):
            chunks.append("\n".join(current))
        current = [*list(header_lines), row]

    if len(current) > len(header_lines):
        chunks.append("\n".join(current))

    if not chunks:
        for row in rows:
            chunks.append("\n".join([*header_lines, row]))

    return chunks
