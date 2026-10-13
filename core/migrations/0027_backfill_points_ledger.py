"""Backfill the points ledger from existing submissions (formula v1).

Self-contained copy of the v1 formula so this migration stays valid if
core/points.py changes later. Reversible: reverse deletes all ledger rows.
"""
from django.db import migrations

BASE_POINTS = {"easy": 10, "medium": 25, "hard": 50}
FAILED = ("wrong_answer", "tle", "compile_error", "runtime_error")


def backfill(apps, schema_editor):
    Submission = apps.get_model("core", "Submission")
    PointsLedger = apps.get_model("core", "PointsLedger")
    seen = set()
    rows = []
    accepted = (
        Submission.objects.filter(status="accepted")
        .select_related("question", "subquestion")
        .order_by("submitted_at", "id")
    )
    for sub in accepted.iterator():
        key = (sub.student_id, sub.question_id, sub.subquestion_id)
        if key in seen:
            continue
        seen.add(key)
        difficulty = sub.subquestion.difficulty if sub.subquestion_id else sub.question.difficulty
        wrong = Submission.objects.filter(
            student_id=sub.student_id,
            question_id=sub.question_id,
            subquestion_id=sub.subquestion_id,
            status__in=FAILED,
            submitted_at__lt=sub.submitted_at,
        ).count()
        base = BASE_POINTS.get(difficulty, 10)
        decay = min(wrong, 5) * 10
        rows.append(
            PointsLedger(
                student_id=sub.student_id,
                question_id=sub.question_id,
                subquestion_id=sub.subquestion_id,
                submission_id=sub.id,
                difficulty=difficulty,
                base_points=base,
                wrong_attempts=wrong,
                decay_percent=decay,
                points=round(base * (100 - decay) / 100),
                formula_version=1,
                awarded_at=sub.judged_at or sub.submitted_at,
            )
        )
    PointsLedger.objects.bulk_create(rows, batch_size=500)


def unbackfill(apps, schema_editor):
    apps.get_model("core", "PointsLedger").objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [("core", "0026_points_ledger")]
    operations = [migrations.RunPython(backfill, unbackfill)]
