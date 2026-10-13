"""T5: server-authoritative points ledger and ranking."""
from datetime import timedelta
from unittest.mock import patch

from django.core.management import call_command
from django.db import IntegrityError
from django.test import TestCase
from django.utils import timezone

from .models import Course, Module, PointsLedger, Question, Submission, User
from .points import BASE_POINTS, award_points, compute_points, rank_tier


class ComputePointsTests(TestCase):
    def test_base_values(self):
        self.assertEqual(BASE_POINTS, {"easy": 10, "medium": 25, "hard": 50})
        for diff, base in BASE_POINTS.items():
            self.assertEqual(compute_points(diff, 0)["points"], base)

    def test_decay_ten_percent_per_wrong_attempt(self):
        self.assertEqual(compute_points("hard", 1)["points"], 45)
        self.assertEqual(compute_points("hard", 3)["points"], 35)

    def test_decay_floored_at_half(self):
        self.assertEqual(compute_points("hard", 5)["points"], 25)
        self.assertEqual(compute_points("hard", 40)["points"], 25)
        self.assertEqual(compute_points("hard", 40)["penalised_attempts"], 5)

    def test_unknown_difficulty_treated_as_easy(self):
        self.assertEqual(compute_points("bogus", 0)["points"], 10)

    def test_tiers(self):
        self.assertEqual(rank_tier(0)["name"], "Bronze")
        self.assertEqual(rank_tier(100)["name"], "Silver")
        self.assertEqual(rank_tier(300)["name"], "Gold")
        self.assertEqual(rank_tier(700)["name"], "Platinum")
        self.assertIsNone(rank_tier(700)["next_name"])
        self.assertEqual(rank_tier(150)["points_to_next"], 150)


class AwardPointsTests(TestCase):
    def setUp(self):
        self.student = User.objects.create_user("s1", password="pw", role=User.Role.STUDENT)
        self.module = Module.objects.create(name="M", course=Course.objects.create(name="C", slug="c"))
        self.q = Question.objects.create(module=self.module, title="Q", slug="q", description="d", difficulty="medium")

    def _sub(self, status, **kw):
        return Submission.objects.create(student=self.student, question=self.q, code="x", status=status, **kw)

    def test_first_accept_awards_base(self):
        row = award_points(self._sub("accepted", score=100))
        self.assertEqual(row.points, 25)
        self.assertEqual(row.wrong_attempts, 0)

    def test_wrong_attempts_before_accept_decay(self):
        self._sub("wrong_answer")
        self._sub("compile_error")
        self._sub("internal_error")  # infrastructure failure: not the student's fault
        row = award_points(self._sub("accepted", score=100))
        self.assertEqual(row.wrong_attempts, 2)
        self.assertEqual(row.points, 20)  # 25 * 0.8

    def test_resolve_awards_nothing(self):
        award_points(self._sub("accepted", score=100))
        self.assertIsNone(award_points(self._sub("accepted", score=100)))
        self.assertEqual(PointsLedger.objects.filter(student=self.student).count(), 1)

    def test_rejudge_same_submission_is_idempotent(self):
        sub = self._sub("accepted", score=100)
        award_points(sub)
        award_points(sub)
        self.assertEqual(PointsLedger.objects.count(), 1)

    def test_non_accepted_awards_nothing(self):
        self.assertIsNone(award_points(self._sub("wrong_answer", score=80)))

    def test_db_constraint_blocks_duplicate_award(self):
        award_points(self._sub("accepted", score=100))
        other = self._sub("accepted", score=100)
        with self.assertRaises(IntegrityError):
            PointsLedger.objects.create(
                student=self.student, question=self.q, submission=other, difficulty="medium",
                base_points=25, wrong_attempts=0, decay_percent=0, points=25,
            )

    @patch("core.hint_service._enqueue_llm_generation")
    def test_hints_do_not_reduce_points(self, _enqueue):
        """Explicit requirement: unlocking hints must never deduct points."""
        from .hint_service import unlock_hint_for_question
        for _ in range(3):
            unlock_hint_for_question(self.student, self.q)
        self.assertEqual(self.student.unlocked_hints.filter(question=self.q).count(), 3)
        row = award_points(self._sub("accepted", score=100))
        self.assertEqual(row.points, 25)

    @patch("core.services.run_code", return_value={"status_id": 3, "status": "Accepted", "stdout": "3"})
    def test_evaluate_submission_awards_server_side(self, _run):
        from .models import TestCase as QTestCase
        from .services import evaluate_submission
        QTestCase.objects.create(question=self.q, stdin="1 2", expected_output="3")
        sub = self._sub("pending")
        evaluate_submission(sub.pk)
        self.assertEqual(PointsLedger.objects.get(submission=sub).points, 25)


class LeaderboardTests(TestCase):
    def setUp(self):
        self.module = Module.objects.create(name="M", course=Course.objects.create(name="C", slug="c"))
        self.qs = {
            d: Question.objects.create(module=self.module, title=d, slug=d, description="d", difficulty=d)
            for d in ("easy", "medium", "hard")
        }

    def _solve(self, user, diff, wrong=0):
        for _ in range(wrong):
            Submission.objects.create(student=user, question=self.qs[diff], code="x", status="wrong_answer")
        award_points(Submission.objects.create(student=user, question=self.qs[diff], code="x", status="accepted", score=100))

    def test_resubmission_farming_does_not_inflate_score(self):
        u = User.objects.create_user("farmer", password="pw", role=User.Role.STUDENT)
        for _ in range(5):
            self._solve(u, "easy")
        self.client.force_login(u)
        row = self.client.get("/leaderboard/").context["rows"][0]
        self.assertEqual(row["points"], 10)
        self.assertEqual(row["solved"], 1)

    def test_tiebreak_on_hard_solved(self):
        a = User.objects.create_user("a", password="pw", role=User.Role.STUDENT)
        b = User.objects.create_user("b", password="pw", role=User.Role.STUDENT)
        # a: one hard with 5 wrong (25 pts) ; b: one medium (25 pts)
        self._solve(a, "hard", wrong=5)
        self._solve(b, "medium")
        self.client.force_login(b)
        rows = self.client.get("/leaderboard/").context["rows"]
        self.assertEqual([r["student"].username for r in rows[:2]], ["a", "b"])
        self.assertEqual(rows[0]["rank"], 1)

    def test_own_row_shown_with_breakdown(self):
        u = User.objects.create_user("me", password="pw", role=User.Role.STUDENT)
        self._solve(u, "medium", wrong=2)
        self.client.force_login(u)
        ctx = self.client.get("/leaderboard/").context
        self.assertEqual(ctx["me"]["points"], 20)
        self.assertEqual(ctx["me"]["tier"]["name"], "Bronze")
        self.assertEqual(len(ctx["my_ledger"]), 1)


class RecalculatePointsCommandTests(TestCase):
    def test_rebuild_from_history(self):
        module = Module.objects.create(name="M")
        q = Question.objects.create(module=module, title="Q", slug="q", description="d", difficulty="hard")
        u = User.objects.create_user("h", password="pw", role=User.Role.STUDENT)
        now = timezone.now()
        w = Submission.objects.create(student=u, question=q, code="x", status="wrong_answer")
        a1 = Submission.objects.create(student=u, question=q, code="x", status="accepted", score=100)
        a2 = Submission.objects.create(student=u, question=q, code="x", status="accepted", score=100)
        Submission.objects.filter(pk=w.pk).update(submitted_at=now - timedelta(minutes=3))
        Submission.objects.filter(pk=a1.pk).update(submitted_at=now - timedelta(minutes=2))
        Submission.objects.filter(pk=a2.pk).update(submitted_at=now - timedelta(minutes=1))
        call_command("recalculate_points")
        row = PointsLedger.objects.get(student=u)
        self.assertEqual(row.submission_id, a1.pk)
        self.assertEqual(row.points, 45)
        call_command("recalculate_points")  # idempotent
        self.assertEqual(PointsLedger.objects.count(), 1)
