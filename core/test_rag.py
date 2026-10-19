"""T7: RAG question generation — chunking, retrieval, validation pipeline, async job API."""
import hashlib
import json
import math
import re
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from .models import GenerationJob, Module, Question, RagChunk, RagDocument, User
from .rag import generate, retrieval
from .rag.chunking import TARGET_CHARS, chunk_markdown


def fake_embed(texts, kind="document"):
    """Deterministic bag-of-words vectors (64-d) so similarity tracks word overlap."""
    out = []
    for text in texts:
        vec = [0.0] * 64
        for word in re.findall(r"[a-z]{3,}", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


STACK_DOC = """# Balanced Brackets
**Topic**: Stack

## Problem Statement
Given a string of brackets, decide whether every opening bracket is closed by the matching bracket in the right order.
Brackets are round, square and curly. A closing bracket must match the most recent unmatched opening bracket.

## Constraints
- 1 <= length <= 10^4

```python
def balanced(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    seen = []
    for ch in s:
        if ch in pairs.values():
            seen.append(ch)
        elif not seen or seen.pop() != pairs[ch]:
            return False
    return not seen
```
"""

GRID_DOC = """# Count Islands
**Topic**: Graph

## Problem Statement
Given a grid of land and water cells, count connected groups of land cells. Cells connect horizontally and vertically.
"""


class ChunkingTests(TestCase):
    def test_code_blocks_never_split(self):
        long_code = "```c\n" + "\n".join(f"int v{i} = {i};" for i in range(400)) + "\n```"
        md = "# T\n\nIntro paragraph.\n\n" + long_code + "\n\nAfter the code."
        chunks = chunk_markdown(md, title="T")
        code_chunks = [c for c in chunks if "```" in c.text]
        self.assertEqual(len(code_chunks), 1)
        self.assertEqual(code_chunks[0].text.count("```"), 2)
        self.assertIn("int v399 = 399;", code_chunks[0].text)

    def test_small_sections_are_merged_and_large_ones_split(self):
        md = "# Doc\n\n## A\n\nshort\n\n## B\n\nalso short\n\n## C\n\n" + "\n\n".join("word " * 60 for _ in range(12))
        chunks = chunk_markdown(md, title="Doc")
        self.assertIn("## B", chunks[0].text)  # tiny sections kept together
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(c.text) <= 2 * TARGET_CHARS for c in chunks))

    def test_overlap_carries_previous_paragraph(self):
        paras = [f"Paragraph {i} " + "text " * 50 for i in range(10)]
        chunks = chunk_markdown("# D\n\n" + "\n\n".join(paras), title="D")
        self.assertGreater(len(chunks), 1)
        last_of_first = chunks[0].text.split("\n\n")[-1]
        self.assertTrue(chunks[1].text.startswith(last_of_first))


@patch("core.rag.retrieval.embed", side_effect=fake_embed)
class RetrievalTests(TestCase):
    def setUp(self):
        retrieval.upsert_document("corpus", "stack/brackets.md", "Balanced Brackets", "stack", STACK_DOC)
        retrieval.upsert_document("corpus", "graph/islands.md", "Count Islands", "graph bfs dfs", GRID_DOC)

    def test_empty_index_raises_clear_error(self, _embed):
        with self.assertRaises(retrieval.RetrievalError) as ctx:
            retrieval.search("brackets")
        self.assertIn("rag_index", str(ctx.exception))

    def test_search_returns_relevant_chunks_with_version_tag(self, _embed):
        retrieval.embed_pending()
        chunk = RagChunk.objects.first()
        self.assertEqual(chunk.embed_model, retrieval.EMBED_MODEL)
        self.assertEqual(chunk.embed_dim, 64)
        hits = retrieval.search("matching brackets opening closing", k=2)
        self.assertEqual(hits[0].document.title, "Balanced Brackets")

    def test_stale_model_chunks_are_ignored_until_reembedded(self, _embed):
        retrieval.embed_pending()
        RagChunk.objects.update(embed_model="old-model")
        with self.assertRaises(retrieval.RetrievalError):
            retrieval.search("brackets")
        self.assertEqual(retrieval.stale_chunks().count(), RagChunk.objects.count())
        retrieval.embed_pending()
        self.assertTrue(retrieval.search("brackets"))

    def test_duplicate_chunks_deduplicated(self, _embed):
        retrieval.upsert_document("corpus", "stack/brackets-copy.md", "Balanced Brackets", "stack", STACK_DOC)
        retrieval.embed_pending()
        hits = retrieval.search("brackets", k=4)
        self.assertEqual(len({h.content_hash for h in hits}), len(hits))

    def test_unchanged_document_not_rechunked(self, _embed):
        doc, changed = retrieval.upsert_document("corpus", "stack/brackets.md", "Balanced Brackets", "stack", STACK_DOC)
        self.assertFalse(changed)


def good_candidate(chunk_ref, **overrides):
    q = {
        "title": "Editor Bracket Highlighter",
        "statement": "A code editor wants to warn about unmatched brackets. Read one line containing only round, square "
                     "and curly brackets and print YES when each one pairs up with a partner of the same kind in a "
                     "properly nested way, otherwise print NO.",
        "input_format": "One line with the bracket string.",
        "output_format": "YES or NO.",
        "constraints": ["1 <= length <= 10^4"],
        "difficulty": "easy",
        "difficulty_rationale": "single pass with one idea",
        "starter_code": "s = input().strip()\n# TODO: decide whether s is balanced\nprint('NO')\n",
        "reference_solution": "s = input().strip()\npairs = {')': '(', ']': '[', '}': '{'}\nseen = []\nok = True\n"
                              "for ch in s:\n    if ch in '([{':\n        seen.append(ch)\n"
                              "    elif not seen or seen.pop() != pairs[ch]:\n        ok = False\n        break\n"
                              "print('YES' if ok and not seen else 'NO')\n",
        "tests": [
            {"input": "()", "expected_output": "YES"},
            {"input": "([)]", "expected_output": "NO"},
            {"input": "{[]}", "expected_output": "YES"},
            {"input": "(", "expected_output": "NO"},
            {"input": "]", "expected_output": "NO"},
        ],
        "citations": [chunk_ref],
    }
    q.update(overrides)
    return q


def python_runner(language, source_code, stdin="", expected_output="", time_limit=2.0, memory_limit_kb=0):
    """Stand-in for the Docker sandbox: actually executes the Python code (tests only)."""
    import contextlib
    import io

    if "TODO" in source_code:
        return {"status_id": 3, "status": "Accepted", "stdout": "NO"}
    out = io.StringIO()
    lines = iter(stdin.splitlines())
    with contextlib.redirect_stdout(out):
        exec(source_code, {"input": lambda *_: next(lines)})  # noqa: S102 - test fixture only
    return {"status_id": 3, "status": "Accepted", "stdout": out.getvalue()}


@patch("core.rag.retrieval.embed", side_effect=fake_embed)
class GenerationPipelineTests(TestCase):
    def setUp(self):
        retrieval.upsert_document("corpus", "stack/brackets.md", "Balanced Brackets", "stack", STACK_DOC)
        with patch("core.rag.retrieval.embed", side_effect=fake_embed):
            retrieval.embed_pending()
        self.chunks = list(RagChunk.objects.select_related("document"))
        self.ref = self.chunks[0].ref

    def run_pipeline(self, responses, **kw):
        raws = [(json.dumps(r) if isinstance(r, dict) else r, {"eval_count": 500}) for r in responses]
        with patch("core.rag.generate._call_model", side_effect=raws):
            return generate.generate_questions("stack", difficulty="easy", language="python", count=1,
                                               runner=python_runner, **kw)

    def test_valid_candidate_accepted_with_report(self, _e):
        result = self.run_pipeline([good_candidate(self.ref)])
        self.assertEqual(len(result["accepted"]), 1)
        report = result["accepted"][0]["validation"]
        self.assertIn("difficulty_check", report)
        self.assertTrue(result["sources"])

    def test_fabricated_citation_rejected_then_retry_succeeds(self, _e):
        result = self.run_pipeline([good_candidate("D999C9"), good_candidate(self.ref)])
        self.assertEqual(len(result["accepted"]), 1)
        self.assertEqual(result["rejected"][0]["stage"], "citation")

    def test_wrong_expected_output_rejected(self, _e):
        bad = good_candidate(self.ref)
        bad["tests"][0]["expected_output"] = "NO"
        result = self.run_pipeline([bad, bad])
        self.assertEqual(result["accepted"], [])
        self.assertTrue(all(r["stage"] == "execution" for r in result["rejected"]))

    def test_wrong_hidden_output_is_corrected_from_execution(self, _e):
        cand = good_candidate(self.ref)
        cand["tests"][4]["expected_output"] = "YES"  # hidden test, model guessed wrong (']' is unbalanced)
        result = self.run_pipeline([cand])
        self.assertEqual(len(result["accepted"]), 1)
        q = result["accepted"][0]
        self.assertEqual(q["tests"][4]["expected_output"], "NO")
        self.assertEqual(q["validation"]["hidden_outputs_corrected"], 1)

    def test_mostly_wrong_hidden_outputs_rejected(self, _e):
        cand = good_candidate(self.ref)
        for t in cand["tests"][2:]:
            t["expected_output"] = "MAYBE"
        result = self.run_pipeline([cand, cand])
        self.assertEqual(result["accepted"], [])
        self.assertIn("disagreed", result["rejected"][0]["detail"])

    def test_retry_sends_compact_feedback_not_previous_answer(self, _e):
        bad = good_candidate("D999C9")
        calls = []

        def fake(messages):
            calls.append(messages)
            return (json.dumps(bad if len(calls) == 1 else good_candidate(self.ref)), {})

        with patch("core.rag.generate._call_model", side_effect=fake):
            generate.generate_questions("stack", difficulty="easy", language="python", count=1, runner=python_runner)
        retry = calls[1]
        self.assertFalse(any(m["role"] == "assistant" for m in retry))
        self.assertIn("was rejected", retry[-1]["content"])
        self.assertEqual(retry[:2], calls[0][:2])

    def test_wrong_language_rejected(self, _e):
        c_code = "#include <stdio.h>\nint main(void){ return 0; }\n"
        result = self.run_pipeline([good_candidate(self.ref, reference_solution=c_code)] * 2)
        self.assertEqual(result["accepted"], [])
        self.assertEqual(result["rejected"][0]["stage"], "language")

    def test_starter_code_that_contains_solution_rejected(self, _e):
        cand = good_candidate(self.ref)
        cand["starter_code"] = cand["reference_solution"]
        result = self.run_pipeline([cand, cand])
        self.assertEqual(result["rejected"][0]["stage"], "starter")

    def test_malformed_json_rejected(self, _e):
        result = self.run_pipeline(["Here is your question: ...", "{not json"])
        self.assertEqual(result["accepted"], [])
        self.assertEqual(result["rejected"][0]["stage"], "schema")

    def test_ungrounded_statement_rejected(self, _e):
        off_topic = good_candidate(
            self.ref, title="Weather Average",
            statement="Compute the average rainfall over several cities across a whole season, and print it rounded."
        )
        result = self.run_pipeline([off_topic, off_topic])
        self.assertEqual(result["rejected"][0]["stage"], "grounding")

    def test_restated_source_problem_rejected(self, _e):
        copied = good_candidate(
            self.ref, title="Balanced Brackets",
            statement="Given a string of brackets, decide whether every opening bracket is closed by the matching "
                      "bracket in the right order. Brackets are round, square and curly.",
        )
        result = self.run_pipeline([copied, copied])
        self.assertEqual(result["accepted"], [])
        self.assertEqual(result["rejected"][0]["stage"], "novelty")

    def test_duplicate_of_existing_question_rejected(self, _e):
        cand = good_candidate(self.ref)
        Question.objects.create(module=Module.objects.create(name="M"), title=cand["title"], slug="dup",
                                description=cand["statement"])
        result = self.run_pipeline([cand, cand])
        self.assertEqual(result["accepted"], [])
        self.assertEqual(result["rejected"][0]["stage"], "duplicate")

    def test_mixed_difficulty_plan(self, _e):
        self.assertEqual(generate.difficulty_plan("mixed", 3), ["easy", "medium", "hard"])
        self.assertEqual(generate.difficulty_plan("hard", 2), ["hard", "hard"])

    def test_time_budget_stops_cleanly_and_keeps_accepted(self, _e):
        clock = [1000.0]

        def fake(messages):
            clock[0] += 400  # each CPU generation takes a while
            return (json.dumps(good_candidate(self.ref, title=f"Bracket Checker {clock[0]:.0f}")), {})

        from types import SimpleNamespace
        with patch("core.rag.generate.time", SimpleNamespace(monotonic=lambda: clock[0])), \
                patch("core.rag.generate.GEN_TIMEOUT", 600), \
                patch("core.rag.generate._call_model", side_effect=fake), \
                patch("core.rag.generate.check_duplicate", return_value=[0.0]):
            result = generate.generate_questions("stack", difficulty="easy", language="python", count=3,
                                                 runner=python_runner, deadline=1700.0)
        # Call 1 starts at 1000 (+600 fits by 1700); call 2 would start at 1400 and could run past it.
        self.assertEqual(len(result["accepted"]), 1)
        self.assertEqual(result["rejected"][-1]["stage"], "time")

    def test_unexpected_error_keeps_accepted_questions(self, _e):
        calls = []

        def runner(*args, **kwargs):
            calls.append(1)
            if len(calls) > 6:  # first question's 5 tests + starter compile succeed
                raise OSError("docker went away")
            return python_runner(*args, **kwargs)

        cands = [good_candidate(self.ref), good_candidate(self.ref, title="Second Bracket Task")]
        with patch("core.rag.generate._call_model", side_effect=[(json.dumps(c), {}) for c in cands]), \
                patch("core.rag.generate.check_duplicate", return_value=[0.0]):
            result = generate.generate_questions("stack", difficulty="easy", language="python", count=2, runner=runner)
        self.assertEqual(len(result["accepted"]), 1)
        self.assertEqual(result["rejected"][-1]["stage"], "internal")

    def test_citations_must_be_a_list(self, _e):
        cand = good_candidate(self.ref)
        cand["citations"] = self.ref
        result = self.run_pipeline([cand, cand])
        self.assertEqual(result["rejected"][0]["stage"], "schema")

    def test_model_down_is_reported_not_raised(self, _e):
        import requests
        with patch("core.rag.generate._call_model", side_effect=requests.ConnectionError("down")):
            result = generate.generate_questions("stack", count=1, runner=python_runner)
        self.assertEqual(result["accepted"], [])
        self.assertEqual(result["rejected"][0]["stage"], "model")


class AgentApiTests(TestCase):
    def setUp(self):
        self.fac = User.objects.create_user("fac", password="pw", role=User.Role.FACULTY)
        self.module = Module.objects.create(name="Stacks")
        self.client.force_login(self.fac)

    @patch("core.views.generate_questions_task.delay")
    def test_generate_queues_job_and_limits_concurrency(self, delay):
        r = self.client.post("/api/faculty-agent/generate/", data=json.dumps({"topic": "stack", "language": "python"}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 202)
        delay.assert_called_once_with(r.json()["job_id"])
        again = self.client.post("/api/faculty-agent/generate/", data=json.dumps({"topic": "stack"}),
                                 content_type="application/json")
        self.assertEqual(again.status_code, 429)

    def test_generate_validates_input(self):
        for body in ({"topic": ""}, {"topic": "x", "language": "rust"}, {"topic": "x", "difficulty": "insane"}):
            r = self.client.post("/api/faculty-agent/generate/", data=json.dumps(body), content_type="application/json")
            self.assertEqual(r.status_code, 400, body)

    def _done_job(self, owner=None):
        return GenerationJob.objects.create(
            created_by=owner or self.fac, status=GenerationJob.Status.DONE,
            params={"language": "python"},
            result={"accepted": [dict(good_candidate("D1C0"), language="python", validation={})], "rejected": [],
                    "sources": [{"ref": "D1C0", "title": "Balanced Brackets"}]},
        )

    def test_save_from_job_creates_inactive_question_with_server_side_tests(self):
        job = self._done_job()
        r = self.client.post("/api/faculty-agent/add-question/", data=json.dumps(
            {"job_id": job.pk, "index": 0, "module_id": self.module.pk, "test_cases": [{"input": "x", "expected_output": "y"}]}
        ), content_type="application/json")
        self.assertEqual(r.status_code, 200, r.content)
        q = Question.objects.get(pk=r.json()["question_id"])
        self.assertFalse(q.is_active)  # pending faculty review
        self.assertEqual(q.language_id, 71)
        self.assertEqual(q.test_cases.count(), 5)
        self.assertFalse(q.test_cases.filter(stdin="x").exists())  # client-sent tests ignored
        self.assertEqual(q.test_cases.filter(is_sample=True).count(), 2)
        dup = self.client.post("/api/faculty-agent/add-question/", data=json.dumps(
            {"job_id": job.pk, "index": 0, "module_id": self.module.pk}), content_type="application/json")
        self.assertEqual(dup.status_code, 409)

    def test_job_is_private_to_its_owner(self):
        other = User.objects.create_user("other", password="pw", role=User.Role.FACULTY)
        job = self._done_job(owner=other)
        self.assertEqual(self.client.get(f"/api/faculty-agent/jobs/{job.pk}/").status_code, 403)
        r = self.client.post("/api/faculty-agent/add-question/", data=json.dumps(
            {"job_id": job.pk, "index": 0, "module_id": self.module.pk}), content_type="application/json")
        self.assertEqual(r.status_code, 404)

    def test_students_cannot_use_agent(self):
        student = User.objects.create_user("st", password="pw", role=User.Role.STUDENT)
        self.client.force_login(student)
        r = self.client.post("/api/faculty-agent/generate/", data=json.dumps({"topic": "stack"}), content_type="application/json")
        self.assertEqual(r.status_code, 403)

    def test_only_llm_job_questions_can_be_saved(self):
        r = self.client.post("/api/faculty-agent/add-question/", data=json.dumps(
            {"module_id": self.module.pk, "title": "T", "description": "D"}), content_type="application/json")
        self.assertEqual(r.status_code, 400)
        self.assertFalse(Question.objects.exists())

    @patch("core.views.generate_questions_task.delay")
    def test_stale_job_does_not_block_new_generation(self, delay):
        import datetime
        from django.utils import timezone
        stuck = GenerationJob.objects.create(created_by=self.fac, status=GenerationJob.Status.RUNNING,
                                             started_at=timezone.now() - datetime.timedelta(hours=3))
        r = self.client.post("/api/faculty-agent/generate/", data=json.dumps({"topic": "stack", "language": "python"}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 202)
        stuck.refresh_from_db()
        self.assertEqual(stuck.status, GenerationJob.Status.FAILED)
        fresh = GenerationJob.objects.create(created_by=self.fac, status=GenerationJob.Status.RUNNING, started_at=timezone.now())
        self.assertEqual(self.client.get(f"/api/faculty-agent/jobs/{fresh.pk}/").json()["status"], "running")

    def test_topics_listed_without_index(self):
        topics = self.client.get("/api/faculty-agent/topics/").json()["topics"]
        self.assertTrue(topics)  # bundled corpus folders, no template generator involved

    @patch("core.views.embed_document_task.delay")
    def test_upload_markdown_creates_document(self, delay):
        f = SimpleUploadedFile("notes.md", STACK_DOC.encode() * 2, content_type="text/markdown")
        r = self.client.post("/api/faculty-agent/upload/", data={"file": f, "topic": "stack"})
        self.assertEqual(r.status_code, 200, r.content)
        doc = RagDocument.objects.get(pk=r.json()["document_id"])
        self.assertEqual(doc.source, "upload")
        self.assertEqual(doc.uploaded_by, self.fac)
        self.assertGreater(doc.chunks.count(), 0)
        delay.assert_called_once_with(doc.pk)

    def test_upload_rejects_bad_files(self):
        exe = SimpleUploadedFile("x.exe", b"MZ" * 500)
        self.assertEqual(self.client.post("/api/faculty-agent/upload/", data={"file": exe}).status_code, 400)
        tiny = SimpleUploadedFile("x.txt", b"too short")
        self.assertEqual(self.client.post("/api/faculty-agent/upload/", data={"file": tiny}).status_code, 400)

    def test_broken_generate_page_redirects(self):
        r = self.client.get("/faculty/questions/generate/")
        self.assertEqual(r.status_code, 302)
