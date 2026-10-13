"""
Embedding index and hybrid retrieval over RagChunk.

Embeddings come from Ollama (`nomic-embed-text`, 768-d) on CPU. Every chunk
stores the embedding model, dimension and EMBED_VERSION; search only uses
chunks embedded with the current model/version, and `manage.py rag_index`
re-embeds anything stale. Retrieval = cosine similarity blended with a
lexical score (helps exact terms like "palindrome" or "BST"), then
de-duplicated by content hash and near-identical embeddings.
"""
import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path

import requests
from django.conf import settings
from django.db import transaction

from ..models import RagChunk, RagDocument
from .chunking import chunk_markdown

EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", os.environ.get("OLLAMA_URL", "http://elab-ollama:11434/api/chat").replace("/api/chat", "/api/embed"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_VERSION = 1  # bump when chunking or embedding input format changes
EMBED_TIMEOUT = 60
CORPUS_DIR = Path(settings.BASE_DIR) / "data" / "DSA_Topics"
SEMANTIC_WEIGHT = 0.75
NEAR_DUPLICATE = 0.97


class RetrievalError(RuntimeError):
    pass


# ── Embeddings ────────────────────────────────────────────────────────────

def embed(texts, kind="document"):
    """Embed a list of strings. nomic-embed-text expects task prefixes."""
    prefix = "search_document: " if kind == "document" else "search_query: "
    payload = {"model": EMBED_MODEL, "input": [prefix + t[:6000] for t in texts]}
    if os.environ.get("OLLAMA_NUM_GPU") is not None:
        payload["options"] = {"num_gpu": int(os.environ["OLLAMA_NUM_GPU"])}
    try:
        resp = requests.post(EMBED_URL, json=payload, timeout=EMBED_TIMEOUT)
        resp.raise_for_status()
        vectors = resp.json().get("embeddings")
    except (requests.RequestException, ValueError) as exc:
        raise RetrievalError(f"embedding service unavailable: {exc}") from exc
    if not vectors or len(vectors) != len(texts):
        raise RetrievalError("embedding service returned an unexpected payload")
    return vectors


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _embed_input(chunk_heading, text):
    return f"{chunk_heading}\n{text}"


# ── Indexing ──────────────────────────────────────────────────────────────

def _topic_from_dir(name):
    return re.sub(r"^\d+_", "", name).replace("_", " ").lower()


def _title_from_markdown(text, fallback):
    m = re.search(r"^#\s+(?:\d+\.\s*)?(.*)", text, re.MULTILINE)
    return (m.group(1).strip() if m else fallback)[:255]


def upsert_document(source, path, title, topic, text, uploaded_by=None):
    """Create/refresh a document and its chunks (embeddings are filled by embed_pending)."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    doc = RagDocument.objects.filter(source=source, path=path).first()
    if doc and doc.content_hash == digest:
        return doc, False
    chunks = chunk_markdown(text, title=title)
    if not chunks:
        raise RetrievalError(f"no usable text in {path}")
    with transaction.atomic():
        if doc is None:
            doc = RagDocument.objects.create(source=source, path=path, title=title, topic=topic,
                                             content_hash=digest, uploaded_by=uploaded_by)
        else:
            doc.title, doc.topic, doc.content_hash = title, topic, digest
            doc.save(update_fields=["title", "topic", "content_hash"])
            doc.chunks.all().delete()
        RagChunk.objects.bulk_create([
            RagChunk(document=doc, index=i, heading=c.heading, text=c.text, content_hash=c.content_hash)
            for i, c in enumerate(chunks)
        ])
    return doc, True


def index_corpus(corpus_dir=CORPUS_DIR):
    """Load the bundled DSA markdown corpus into RagDocument/RagChunk. Returns (seen, changed)."""
    seen = changed = 0
    for topic_dir in sorted(Path(corpus_dir).iterdir()):
        if not topic_dir.is_dir():
            continue
        topic = _topic_from_dir(topic_dir.name)
        for md in sorted(topic_dir.glob("*.md")):
            if md.name.lower() == "readme.md":
                continue
            text = md.read_text(encoding="utf-8", errors="ignore")
            rel = str(md.relative_to(corpus_dir))
            _, did_change = upsert_document(RagDocument.Source.CORPUS, rel, _title_from_markdown(text, md.stem), topic, text)
            seen += 1
            changed += int(did_change)
    return seen, changed


def stale_chunks():
    return RagChunk.objects.exclude(embed_model=EMBED_MODEL, embed_version=EMBED_VERSION) | RagChunk.objects.filter(embedding__isnull=True)


def embed_pending(batch_size=32, document=None, progress=None):
    """Embed every chunk that has no embedding or was embedded with another model/version."""
    qs = stale_chunks()
    if document is not None:
        qs = qs.filter(document=document)
    pending = list(qs.order_by("id"))
    done = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        vectors = embed([_embed_input(c.heading, c.text) for c in batch])
        for chunk, vec in zip(batch, vectors):
            chunk.embedding = vec
            chunk.embed_model = EMBED_MODEL
            chunk.embed_dim = len(vec)
            chunk.embed_version = EMBED_VERSION
        RagChunk.objects.bulk_update(batch, ["embedding", "embed_model", "embed_dim", "embed_version"])
        done += len(batch)
        if progress:
            progress(done, len(pending))
    return done


# ── Search ────────────────────────────────────────────────────────────────

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in for and or with on by is are be from that this it as at your you".split())


def _tokens(text):
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _lexical_scores(query, chunks):
    q = Counter(_tokens(query))
    if not q:
        return [0.0] * len(chunks)
    docs = [Counter(_tokens(f"{c.document.topic} {c.heading} {c.text}")) for c in chunks]
    n = len(docs)
    df = Counter(t for d in docs for t in set(d) if t in q)
    scores = []
    for d in docs:
        length = sum(d.values()) or 1
        s = 0.0
        for t in q:
            if d[t]:
                idf = math.log(1 + n / (1 + df[t]))
                s += idf * (d[t] * 2.2) / (d[t] + 1.2 * (0.25 + 0.75 * length / 120))
        scores.append(s)
    top = max(scores) or 1.0
    return [s / top for s in scores]


def search(query, k=4, document_ids=None, topic=None, mode="hybrid"):
    """
    Return up to k distinct chunks for `query`. Raises RetrievalError if the
    index is empty/stale or nothing relevant is found (never silently empty).
    """
    qs = RagChunk.objects.select_related("document").filter(embed_model=EMBED_MODEL, embed_version=EMBED_VERSION)
    if document_ids:
        qs = qs.filter(document_id__in=document_ids)
    if topic:
        topic_qs = qs.filter(document__topic__icontains=topic.lower())
        if topic_qs.exists():
            qs = topic_qs
    chunks = [c for c in qs if c.embedding]
    if not chunks:
        raise RetrievalError(
            "The question-generation index is empty or was built with a different embedding model. "
            "Run: python manage.py rag_index"
        )
    lexical = _lexical_scores(query, chunks) if mode in ("hybrid", "lexical") else [0.0] * len(chunks)
    if mode in ("hybrid", "semantic"):
        qvec = embed([query], kind="query")[0]
        semantic = [cosine(qvec, c.embedding) for c in chunks]
    else:
        semantic = [0.0] * len(chunks)
    if mode == "hybrid":
        scores = [SEMANTIC_WEIGHT * s + (1 - SEMANTIC_WEIGHT) * l for s, l in zip(semantic, lexical)]
    else:
        scores = semantic if mode == "semantic" else lexical
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    picked, hashes = [], set()
    for score, chunk in ranked:
        if score <= 0:
            break
        if chunk.content_hash in hashes:
            continue
        if any(cosine(chunk.embedding, p.embedding) >= NEAR_DUPLICATE for _, p in picked):
            continue
        picked.append((score, chunk))
        hashes.add(chunk.content_hash)
        if len(picked) == k:
            break
    if not picked:
        raise RetrievalError(f"No source material matched '{query}'. Try a broader topic or upload notes on it.")
    return [c for _, c in picked]
