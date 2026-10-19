from config.celery import app

import logging

from .services import evaluate_submission

logger = logging.getLogger(__name__)


@app.task
def evaluate_submission_task(submission_id):
    sub = evaluate_submission(submission_id)
    return sub.pk if sub else submission_id


@app.task(soft_time_limit=60, time_limit=70)
def run_code_task(language, source_code, stdin="", expected_output="", time_limit=2.0, memory_limit_kb=128000):
    """Execute code in the Docker sandbox on a worker (the web container has no Docker socket)."""
    from .sandbox import run_code

    return run_code(language, source_code=source_code, stdin=stdin, expected_output=expected_output,
                    time_limit=time_limit, memory_limit_kb=memory_limit_kb)


@app.task
def generate_submission_hint_task(submission_id):
    from .models import Submission
    from .hint_service import generate_hint_for_submission

    submission = Submission.objects.filter(pk=submission_id).first()
    if submission and submission.status != Submission.Status.ACCEPTED:
        hint = generate_hint_for_submission(submission)
        return hint.pk if hint else None
    return None


@app.task(soft_time_limit=300, time_limit=330)
def generate_hint_set_task(question_id, language, code_digest="-", student_id=None):
    """
    Generate, validate and store three LLM hints for (question, language).
    Falls back silently (students keep the deterministic hint) if the model is
    slow, down, or its output keeps failing the leak guard.
    """
    from . import llm_hints
    from .hint_service import get_hint_set, mark_generation_finished, store_hint_set
    from .models import Question, Submission

    question = Question.objects.filter(pk=question_id, is_active=True).first()
    if not question:
        return None
    existing = get_hint_set(question, language, code_digest)
    if existing:
        # A duplicate job (queued again after the in-flight marker expired):
        # just make sure fallback rows are upgraded, don't spend another CPU run.
        store_hint_set(question, language, code_digest, existing, student_id)
        mark_generation_finished(question, language, code_digest, failed=False)
        return existing
    failed = True
    try:
        student_code = None
        if student_id and code_digest != "-":
            latest = Submission.objects.filter(student_id=student_id, question=question).order_by("-id").first()
            student_code = latest.code if latest and llm_hints.code_hash(latest.code) == code_digest else None
        hints, info = llm_hints.generate_with_llm(question, language, student_code)
        if hints:
            store_hint_set(question, language, code_digest, hints, student_id)
            failed = False
            return hints
        logger.info(f"LLM hints unavailable for Q{question_id} ({language}): {info}")
        return None
    except Exception as exc:
        logger.warning(f"LLM hint generation failed for Q{question_id} ({language}): {exc}", exc_info=True)
        return None
    finally:
        mark_generation_finished(question, language, code_digest, failed)


@app.task
def generate_llm_hint_task(question_id, tier=None):
    """Deprecated per-tier task kept so messages queued before upgrade still run."""
    from .hint_service import pregenerate_hints_for_question
    from .models import Question

    question = Question.objects.filter(pk=question_id, is_active=True).first()
    if question:
        return pregenerate_hints_for_question(question)
    return None


@app.task(soft_time_limit=3600, time_limit=3660)
def generate_questions_task(job_id):
    """Run an LLM question-generation job (T7) on the CPU 'llm' queue."""
    import time

    from django.utils import timezone

    from .models import GenerationJob
    from .rag import generate, retrieval

    job = GenerationJob.objects.filter(pk=job_id).first()
    if not job or job.status not in (GenerationJob.Status.QUEUED, GenerationJob.Status.RUNNING):
        return None
    job.status, job.started_at = GenerationJob.Status.RUNNING, timezone.now()
    job.save(update_fields=["status", "started_at"])

    def progress(msg):
        GenerationJob.objects.filter(pk=job_id).update(progress=msg[:200])

    p = job.params
    try:
        result = generate.generate_questions(
            topic=p.get("topic", ""),
            difficulty=p.get("difficulty", "mixed"),
            language=p.get("language", "python"),
            count=int(p.get("count", 1)),
            custom_prompt=p.get("prompt", ""),
            document_ids=p.get("document_ids") or None,
            progress=progress,
            deadline=time.monotonic() + generate.JOB_TIME_BUDGET,
        )
        job.result = result
        job.status = GenerationJob.Status.DONE
        if not result["accepted"]:
            job.error = "No question passed validation; see the rejection reasons."
    except retrieval.RetrievalError as exc:
        job.status, job.error = GenerationJob.Status.FAILED, str(exc)
    except Exception as exc:  # never leave a job stuck in 'running'
        logger.exception("Question generation job %s failed", job_id)
        job.status, job.error = GenerationJob.Status.FAILED, f"Generation failed ({type(exc).__name__})."
    job.finished_at = timezone.now()
    job.progress = ""
    job.save(update_fields=["status", "result", "error", "finished_at", "progress"])
    return job.status


@app.task(soft_time_limit=900, time_limit=960)
def embed_document_task(document_id):
    from .models import RagDocument
    from .rag import retrieval

    doc = RagDocument.objects.filter(pk=document_id).first()
    if doc:
        retrieval.embed_pending(document=doc)
