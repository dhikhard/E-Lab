"""
Request options shared by every call to the local hint/question model.

Ollama reloads a model whenever a request asks for a different context size
(num_ctx) than the loaded instance, and on CPU a reload of qwen3.5:4b costs
tens of seconds. Hints and question generation therefore use ONE context size,
large enough for generation prompts (system + four source chunks + the JSON
answer); hint prompts simply leave part of it unused.
"""
import os

NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", os.environ.get("QUESTION_GEN_CTX", "6144")))
NUM_THREAD = int(os.environ.get("OLLAMA_NUM_THREAD", "4"))
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "24h")


def apply(payload):
    """Fill in the shared options on an Ollama /api/chat payload (in place) and return it."""
    options = payload.setdefault("options", {})
    options["num_ctx"] = NUM_CTX
    options.setdefault("num_thread", NUM_THREAD)
    if os.environ.get("OLLAMA_NUM_GPU") is not None:
        # 0 forces CPU inference (the production server has no GPU).
        options["num_gpu"] = int(os.environ["OLLAMA_NUM_GPU"])
    if os.environ.get("OLLAMA_THINK", "false").lower() == "false":
        payload["think"] = False  # "thinking" models (qwen3.x): skip hidden reasoning tokens on CPU
    payload["keep_alive"] = KEEP_ALIVE
    return payload
