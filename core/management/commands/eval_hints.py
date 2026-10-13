"""
Run the hint eval set (core/evals/hint_eval_set.json) against the configured
Ollama model and write raw results for manual grading.

    python manage.py eval_hints --out hint_eval_results.json

Uses temporary Question rows inside a transaction that is rolled back.
"""
import json
import time
from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import transaction

from core import llm_hints
from core.models import Module, Question, TestCase

EVAL_SET = Path(__file__).resolve().parents[2] / "evals" / "hint_eval_set.json"
LANGUAGE_IDS = {"c": 50, "cpp": 54, "java": 62, "python": 71}


class Command(BaseCommand):
    help = "Evaluate LLM hint generation on the bundled eval set (no data is kept)."

    def add_arguments(self, parser):
        parser.add_argument("--out", default="hint_eval_results.json")
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **options):
        cases = json.loads(EVAL_SET.read_text())
        if options["limit"]:
            cases = cases[: options["limit"]]
        results = []
        with transaction.atomic():
            module = Module.objects.create(name="hint-eval (temporary)")
            for i, case in enumerate(cases, 1):
                q = Question.objects.create(
                    module=module, title=case["title"], slug=f"eval-{i}", description=case["description"],
                    difficulty=case["difficulty"], language_id=LANGUAGE_IDS[case["language"]],
                )
                for n, ex in enumerate(case["examples"]):
                    inp, _, out = ex.partition("Output:")
                    TestCase.objects.create(question=q, stdin=inp.replace("Input:", "").strip(),
                                            expected_output=out.split("Explanation:")[0].strip(), is_sample=True, order=n)
                if case.get("hidden_expected_output"):
                    TestCase.objects.create(question=q, stdin="hidden", expected_output=case["hidden_expected_output"], is_sample=False)
                started = time.monotonic()
                hints, info = llm_hints.generate_with_llm(q, case["language"])
                row = {
                    "id": case["id"], "language": case["language"], "difficulty": case["difficulty"],
                    "crux_terms": case["crux_terms"], "hints": hints,
                    "fallback_used": hints is None,
                    "attempts": info["attempts"], "guard_reasons": info["guard_reasons"], "errors": info["errors"],
                    "calls": info["calls"], "total_s": round(time.monotonic() - started, 1),
                }
                results.append(row)
                state = "FALLBACK" if hints is None else f"ok ({info['attempts']} attempt(s))"
                self.stdout.write(f"[{i}/{len(cases)}] {case['id']} [{case['language']}] {state} {row['total_s']}s")
            transaction.set_rollback(True)
        Path(options["out"]).write_text(json.dumps({"model": llm_hints.HINT_MODEL, "results": results}, indent=1))
        ok = sum(1 for r in results if r["hints"])
        fired = sum(1 for r in results if r["guard_reasons"])
        self.stdout.write(self.style.SUCCESS(
            f"{ok}/{len(results)} produced valid hints; guard fired on {fired}/{len(results)} cases; written to {options['out']}"
        ))
