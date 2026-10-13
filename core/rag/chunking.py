"""
Structure-aware chunking.

Markdown is split into blocks (headings, paragraphs, lists, fenced code) and
packed into chunks of ~TARGET_CHARS within a section. A fenced code block is
never split, even if it alone exceeds the target. The last prose block of a
chunk is repeated at the start of the next chunk in the same section (overlap)
so a sentence that introduces a code block or example isn't orphaned.

Sizing: ~1,200 characters is ~300 tokens. The bundled DSA documents are
~100 lines each, so a document becomes ~3-6 chunks: statement + I/O format,
examples, constraints/complexity, solution code. Four chunks fit easily in a
4k-token generation prompt on a CPU model, and in nomic-embed-text's window.
"""
import hashlib
import re
from dataclasses import dataclass

TARGET_CHARS = 1200
MAX_CHARS = 2400
OVERLAP_MAX_CHARS = 300


@dataclass
class Chunk:
    heading: str
    text: str
    has_code: bool

    @property
    def content_hash(self):
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _blocks(markdown):
    """Yield (kind, text, heading_level) blocks; kind in heading|code|prose."""
    lines = markdown.replace("\r\n", "\n").split("\n")
    i = 0
    buf = []

    def flush():
        text = "\n".join(buf).strip()
        buf.clear()
        if text and not re.fullmatch(r"[-*_\s]{3,}", text):
            return ("prose", text, 0)
        return None

    while i < len(lines):
        line = lines[i]
        fence = re.match(r"^\s*(```|~~~)", line)
        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        if fence:
            block = flush()
            if block:
                yield block
            marker = fence.group(1)
            code = [line]
            i += 1
            while i < len(lines):
                code.append(lines[i])
                if lines[i].strip().startswith(marker):
                    break
                i += 1
            yield ("code", "\n".join(code).strip(), 0)
        elif heading:
            block = flush()
            if block:
                yield block
            yield ("heading", heading.group(2).strip(), len(heading.group(1)))
        elif not line.strip():
            block = flush()
            if block:
                yield block
        else:
            buf.append(line)
        i += 1
    block = flush()
    if block:
        yield block


def chunk_markdown(markdown, title=""):
    chunks = []
    path = []  # heading stack: (level, text)
    current, size, last_prose = [], 0, None
    chunk_heading = None

    def heading_text():
        parts = ([title] if title else []) + [h for _, h in path]
        return " > ".join(parts)

    def emit():
        nonlocal current, size, chunk_heading
        text = "\n\n".join(current).strip()
        if text:
            chunks.append(Chunk(heading=(chunk_heading or heading_text())[:300], text=text,
                                has_code="```" in text or "~~~" in text))
        current, size, chunk_heading = [], 0, None

    for kind, text, level in _blocks(markdown):
        if kind == "heading":
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, text))
            # A heading starts a new chunk only when the current one is reasonably full;
            # small sections are kept together (with the heading inline) instead of
            # producing many tiny fragments.
            if size >= TARGET_CHARS // 2:
                emit()
                last_prose = None
            if not current:
                chunk_heading = heading_text()
            current.append("#" * level + " " + text)
            size += len(text)
            continue
        if size + len(text) > TARGET_CHARS and size >= TARGET_CHARS // 3:
            carry = last_prose if last_prose and len(last_prose) <= OVERLAP_MAX_CHARS and kind != "code" else None
            emit()
            chunk_heading = heading_text()
            if carry:
                current, size = [carry], len(carry)
        if not current:
            chunk_heading = heading_text()
        current.append(text)
        size += len(text)
        if kind == "prose":
            last_prose = text
    emit()
    return chunks


def chunk_plaintext(text, title=""):
    """Plain text / PDF text: paragraphs become prose blocks, packed the same way."""
    return chunk_markdown(text, title=title)


def extract_pdf_text(file_obj, max_pages=60):
    from pypdf import PdfReader

    reader = PdfReader(file_obj)
    pages = []
    for page in reader.pages[:max_pages]:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)
