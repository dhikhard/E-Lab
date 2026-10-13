from django.core.management.base import BaseCommand
from core.models import Question
from core.hint_service import pregenerate_hints_for_question


class Command(BaseCommand):
    help = "Queue LLM hint generation (all student languages) for questions so hints are ready before a lab."

    def add_arguments(self, parser):
        parser.add_argument("--question-id", type=int, help="Target a specific question ID")
        parser.add_argument("--module-id", type=int, help="Target a specific module ID")
        parser.add_argument("--limit", type=int, default=50, help="Maximum number of questions to process (default 50)")
        parser.add_argument("--force", action="store_true", help="Force regeneration even if already cached")

    def handle(self, *args, **options):
        q_id = options.get("question_id")
        m_id = options.get("module_id")
        limit = options.get("limit")
        force = options.get("force")

        qs = Question.objects.filter(is_active=True)
        if q_id:
            qs = qs.filter(id=q_id)
        elif m_id:
            qs = qs.filter(module_id=m_id)

        questions = list(qs[:limit])
        total = len(questions)
        self.stdout.write(self.style.SUCCESS(f"Queueing LLM hint generation for {total} question(s)..."))

        success_count = 0
        for idx, q in enumerate(questions, 1):
            self.stdout.write(f"[{idx}/{total}] Pre-warming hints for Question #{q.id}: '{q.title}'...")
            try:
                res = pregenerate_hints_for_question(q, force=force)
                summary = ", ".join(f"{lang}: {state}" for lang, state in res.items())
                self.stdout.write(self.style.SUCCESS(f"  ✓ #{q.id}: {summary}"))
                success_count += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  ✗ Failed for #{q.id}: {e}"))

        self.stdout.write(self.style.SUCCESS(f"Done. Queued or found hints for {success_count}/{total} questions."))
