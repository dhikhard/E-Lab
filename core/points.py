"""
Points and ranking (formula version 1).

- Base points by difficulty: easy 10, medium 25, hard 50.
- Awarded once, on the first accepted submission per question (or DBMS
  sub-question). Re-solving awards nothing.
- Each failed attempt before that first accept reduces the award by 10%,
  capped at 5 attempts, so the award never drops below 50% of base.
- Hints never affect points.
- Infrastructure failures (internal_error) and unjudged submissions are not
  counted as failed attempts.
"""
from django.db import IntegrityError, transaction
from django.db.models import Count, Max, Q, Sum

from .models import PointsLedger, Submission

FORMULA_VERSION = 1
BASE_POINTS = {"easy": 10, "medium": 25, "hard": 50}
DECAY_PER_WRONG_ATTEMPT = 10  # percent
MAX_PENALISED_ATTEMPTS = 5  # 5 x 10% = 50% floor

FAILED_STATUSES = (
    Submission.Status.WRONG_ANSWER,
    Submission.Status.TLE,
    Submission.Status.COMPILE_ERROR,
    Submission.Status.RUNTIME_ERROR,
)

# (minimum points, tier name); ascending.
TIERS = [(0, "Bronze"), (100, "Silver"), (300, "Gold"), (700, "Platinum")]


def compute_points(difficulty, wrong_attempts):
    base = BASE_POINTS.get(difficulty, BASE_POINTS["easy"])
    penalised = min(max(wrong_attempts, 0), MAX_PENALISED_ATTEMPTS)
    decay = penalised * DECAY_PER_WRONG_ATTEMPT
    return {
        "base_points": base,
        "penalised_attempts": penalised,
        "decay_percent": decay,
        "points": round(base * (100 - decay) / 100),
    }


def rank_tier(points):
    current = TIERS[0]
    nxt = None
    for i, tier in enumerate(TIERS):
        if points >= tier[0]:
            current = tier
            nxt = TIERS[i + 1] if i + 1 < len(TIERS) else None
    return {
        "name": current[1],
        "next_name": nxt[1] if nxt else None,
        "points_to_next": (nxt[0] - points) if nxt else 0,
    }


def _difficulty_for(submission):
    if submission.subquestion_id:
        return submission.subquestion.difficulty
    return submission.question.difficulty


def _prior_failures(submission):
    return Submission.objects.filter(
        student_id=submission.student_id,
        question_id=submission.question_id,
        subquestion_id=submission.subquestion_id,
        status__in=FAILED_STATUSES,
        submitted_at__lt=submission.submitted_at,
    ).count()


def award_points(submission):
    """
    Award points for an accepted submission. Returns the new ledger row, or
    None if nothing was awarded (not accepted, or already solved). Safe to call
    repeatedly; uniqueness is enforced by DB constraints.
    """
    if submission.status != Submission.Status.ACCEPTED:
        return None
    existing = PointsLedger.objects.filter(
        student_id=submission.student_id,
        question_id=submission.question_id,
        subquestion_id=submission.subquestion_id,
    ).exists()
    if existing:
        return None

    wrong = _prior_failures(submission)
    difficulty = _difficulty_for(submission)
    calc = compute_points(difficulty, wrong)
    try:
        with transaction.atomic():
            return PointsLedger.objects.create(
                student_id=submission.student_id,
                question_id=submission.question_id,
                subquestion_id=submission.subquestion_id,
                submission=submission,
                difficulty=difficulty,
                base_points=calc["base_points"],
                wrong_attempts=wrong,
                decay_percent=calc["decay_percent"],
                points=calc["points"],
                formula_version=FORMULA_VERSION,
                awarded_at=submission.judged_at or submission.submitted_at,
            )
    except IntegrityError:
        # A concurrent judge already awarded this question.
        return None


def recalculate_all():
    """Rebuild the ledger from submission history (first accept per question)."""
    created = 0
    with transaction.atomic():
        PointsLedger.objects.all().delete()
        accepted = (
            Submission.objects.filter(status=Submission.Status.ACCEPTED)
            .select_related("question", "subquestion")
            .order_by("submitted_at", "id")
        )
        for sub in accepted.iterator():
            if award_points(sub):
                created += 1
    return created


def leaderboard_rows(students_qs):
    """Annotated, ranked rows: points desc, then hard solved, medium solved, earliest to reach score."""
    annotated = students_qs.annotate(
        points=Sum("points_ledger__points", default=0),
        solved=Count("points_ledger", distinct=True),
        hard_solved=Count("points_ledger", filter=Q(points_ledger__difficulty="hard"), distinct=True),
        medium_solved=Count("points_ledger", filter=Q(points_ledger__difficulty="medium"), distinct=True),
        last_award=Max("points_ledger__awarded_at"),
    )
    rows = sorted(
        annotated,
        key=lambda s: (
            -s.points,
            -s.hard_solved,
            -s.medium_solved,
            s.last_award.timestamp() if s.last_award else float("inf"),
            s.username.lower(),
        ),
    )
    return [
        {
            "rank": i,
            "student": s,
            "points": s.points,
            "solved": s.solved,
            "hard_solved": s.hard_solved,
            "medium_solved": s.medium_solved,
            "tier": rank_tier(s.points),
        }
        for i, s in enumerate(rows, start=1)
    ]


def student_rank(student):
    """1-based rank of `student` among all students (same ordering as the leaderboard)."""
    from .models import User

    for row in leaderboard_rows(User.objects.filter(role=User.Role.STUDENT)):
        if row["student"].pk == student.pk:
            return row["rank"]
    return None
