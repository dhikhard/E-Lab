"""
Run question generation on specific source documents and record everything
for manual grading. Example:

    python manage.py eval_question_gen --doc 04_Stack/01_valid_parentheses.md:python \
        --count 3 --out question_eval.json

Each --doc is <corpus-relative path>:<language>. Retrieval is restricted to
that one document, mirroring "generate from this study material".
"""
import json
import time
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from core.models import RagDocument
from core.rag import generate


class Command(BaseCommand):
    help = "Evaluate RAG question generation on chosen source documents."

    def add_arguments(self, parser):
        parser.add_argument("--doc", action="append", required=True)
        parser.add_argument("--count", type=int, default=3)
        parser.add_argument("--difficulty", default="mixed")
        parser.add_argument("--out", default="question_eval.json")

    def handle(self, *args, **options):
        runs = []
        for spec in options["doc"]:
            path, _, language = spec.rpartition(":")
            doc = RagDocument.objects.filter(path=path).first()
            if not doc:
                raise CommandError(f"{path} is not indexed; run rag_index first")
            started = time.monotonic()
            self.stdout.write(f"== {doc.title} [{language}] ==")
            result = generate.generate_questions(
                topic=doc.topic or doc.title, difficulty=options["difficulty"], language=language,
                count=options["count"], document_ids=[doc.pk],
                progress=lambda msg: self.stdout.write(f"   {msg}"),
            )
            result.update(document=doc.title, path=path, language=language,
                          wall_s=round(time.monotonic() - started, 1))
            runs.append(result)
            self.stdout.write(f"   accepted {len(result['accepted'])}/{options['count']}, "
                              f"rejected attempts {len(result['rejected'])}, {result['wall_s']} s")
            Path(options["out"]).write_text(json.dumps({"model": generate.GEN_MODEL, "runs": runs}, indent=1))
        self.stdout.write(self.style.SUCCESS(f"Written to {options['out']}"))
