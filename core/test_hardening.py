"""Regression tests for the audit quick wins (F1, F4, F13, F16, F17, F19, F20, F24)."""
import json
from datetime import timedelta
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase
from django.utils import timezone

from .models import AssignedQuestion, Course, Module, ModuleQuestionAssignment, Question, Submission, User


def _make_question(module, title, difficulty="easy", **kwargs):
    return Question.objects.create(
        module=module, title=title, slug=title.lower(), description="d", difficulty=difficulty, **kwargs
    )


class _Base(TestCase):
    def setUp(self):
        self.student = User.objects.create_user("stud", password="pw", role=User.Role.STUDENT)
        self.course = Course.objects.create(name="C Programming", slug="c")
        self.module = Module.objects.create(name="M1", course=self.course)
        self.easy = _make_question(self.module, "Easy")
        self.hard = _make_question(self.module, "Hard", difficulty="hard")
        self.client.force_login(self.student)

    def assign(self, question):
        assignment, _ = ModuleQuestionAssignment.objects.get_or_create(
            student=self.student, module=question.module, difficulty=question.difficulty
        )
        AssignedQuestion.objects.get_or_create(
            assignment=assignment, question=question, defaults={"order": 1, "unlocked_at": timezone.now()}
        )


class SubmissionApiImmutabilityTests(_Base):
    """F1: judged submissions must not be editable or deletable by students."""

    def setUp(self):
        super().setUp()
        self.sub = Submission.objects.create(
            student=self.student, question=self.easy, code="ok", status="accepted", score=100
        )

    def test_patch_is_rejected(self):
        r = self.client.patch(
            f"/api/submissions/{self.sub.id}/",
            data=json.dumps({"question": self.hard.id, "code": "swapped"}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 405)
        self.sub.refresh_from_db()
        self.assertEqual(self.sub.question_id, self.easy.id)
        self.assertEqual(self.sub.code, "ok")

    def test_put_is_rejected(self):
        r = self.client.put(
            f"/api/submissions/{self.sub.id}/",
            data=json.dumps({"question": self.hard.id, "code": "x"}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 405)

    def test_delete_is_rejected(self):
        r = self.client.delete(f"/api/submissions/{self.sub.id}/")
        self.assertEqual(r.status_code, 405)
        self.assertTrue(Submission.objects.filter(pk=self.sub.pk).exists())

    @patch("core.views.evaluate_submission_task.delay")
    def test_create_requires_assignment(self, _delay):
        r = self.client.post(
            "/api/submissions/",
            data=json.dumps({"question": self.hard.id, "code": "x"}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 403)
        self.assertFalse(Submission.objects.filter(question=self.hard).exists())

    @patch("core.views.evaluate_submission_task.delay")
    def test_create_allowed_when_assigned(self, delay):
        self.assign(self.hard)
        r = self.client.post(
            "/api/submissions/",
            data=json.dumps({"question": self.hard.id, "code": "x"}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 201)
        delay.assert_called_once()


class RunCodeThrottleTests(_Base):
    """F4: run_code_api must be rate-limited and gated to assigned questions."""

    @patch("core.views.sandbox_run_code", return_value={"status_id": 3, "status": "Accepted", "stdout": "1"})
    def test_unassigned_question_rejected(self, _run):
        r = self.client.post(
            "/api/run/", data=json.dumps({"question": self.hard.id, "code": "x"}), content_type="application/json"
        )
        self.assertEqual(r.status_code, 403)

    @patch("core.views.sandbox_run_code", return_value={"status_id": 3, "status": "Accepted", "stdout": "1"})
    def test_rapid_runs_throttled(self, _run):
        self.assign(self.easy)
        body = json.dumps({"question": self.easy.id, "code": "x", "custom_input": "1"})
        first = self.client.post("/api/run/", data=body, content_type="application/json")
        second = self.client.post("/api/run/", data=body, content_type="application/json")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)


class LabRecordLatestSubmissionTests(_Base):
    """F13: lab record must print the most recent accepted code."""

    def test_latest_accepted_code_is_used(self):
        old = Submission.objects.create(student=self.student, question=self.easy, code="OLD", status="accepted", score=100)
        Submission.objects.create(student=self.student, question=self.easy, code="NEW", status="accepted", score=100)
        Submission.objects.filter(pk=old.pk).update(submitted_at=timezone.now() - timedelta(days=1))
        r = self.client.get("/lab-record/")
        sub = r.context["module_data"][0]["questions"][0]["submission"]
        self.assertEqual(sub.code, "NEW")


class HintGatingTests(_Base):
    """F16/F17: hint unlock gated to assigned questions; status API doesn't leak locked hints."""

    @patch("core.hint_service.enqueue_hint_generation")
    def test_unlock_unassigned_question_rejected(self, enqueue):
        r = self.client.post(f"/questions/{self.hard.id}/hints/unlock/")
        self.assertEqual(r.status_code, 403)
        enqueue.assert_not_called()

    def test_status_does_not_leak_locked_hint_text(self):
        from django.core.cache import cache
        self.assign(self.easy)
        cache.set(f"elab_theory_hint_{self.easy.id}_1", "SECRET NEXT HINT")
        cache.set(f"elab_llm_kind_{self.easy.id}_1", "llm")
        r = self.client.get(f"/questions/{self.easy.id}/hints/status/")
        self.assertNotIn("SECRET NEXT HINT", r.content.decode())


class CreateHodTests(TestCase):
    """F19: no hard-coded default HOD password."""

    def test_requires_password(self):
        with self.assertRaises(CommandError):
            call_command("create_hod")
        self.assertFalse(User.objects.filter(role=User.Role.HOD).exists())

    def test_uses_supplied_password(self):
        call_command("create_hod", password="a-strong-pass-123")
        hod = User.objects.get(role=User.Role.HOD)
        self.assertTrue(hod.check_password("a-strong-pass-123"))
        self.assertFalse(hod.check_password("hodpassword"))


class AllowedHostsTests(TestCase):
    """F20: ALLOWED_HOSTS is no longer a wildcard."""

    def test_no_wildcard(self):
        from django.conf import settings
        self.assertNotIn("*", settings.ALLOWED_HOSTS)
        self.assertIn("localhost", settings.ALLOWED_HOSTS)


class AgentErrorLeakTests(TestCase):
    """F24: internal exception text is not returned to the client.

    Generation is now an async job (T7), so the error surfaces through the job API.
    """

    @patch("core.rag.generate.generate_questions", side_effect=RuntimeError("db password=hunter2"))
    def test_generate_error_is_generic(self, _gen):
        from .models import GenerationJob
        from .tasks import generate_questions_task

        fac = User.objects.create_user("fac", password="pw", role=User.Role.FACULTY)
        self.client.force_login(fac)
        job = GenerationJob.objects.create(created_by=fac, params={"topic": "array"})
        generate_questions_task.apply(args=[job.pk])
        r = self.client.get(f"/api/faculty-agent/jobs/{job.pk}/")
        self.assertEqual(r.json()["status"], "failed")
        self.assertNotIn("hunter2", r.content.decode())


class SubmissionLanguageTests(_Base):
    """A question's language can't be swapped by sending another language_id."""

    def setUp(self):
        from django.core.cache import cache
        cache.clear()  # submit cooldowns are keyed by ids the test DB reuses
        super().setUp()
        self.assign(self.easy)  # language_id defaults to C (50), single language

    @patch("core.views.evaluate_submission_task.delay")
    def test_api_ignores_other_language_on_single_language_question(self, _delay):
        r = self.client.post("/api/submissions/", data=json.dumps(
            {"question": self.easy.id, "code": "print(1)", "language_id": 71}), content_type="application/json")
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual(Submission.objects.get().language_id, self.easy.language_id)

    @patch("core.views.evaluate_submission_task.delay")
    def test_multi_language_question_accepts_supported_choice_only(self, _delay):
        from django.core.cache import cache

        self.easy.allow_multiple_languages = True
        self.easy.save()
        self.client.post("/api/submissions/", data=json.dumps(
            {"question": self.easy.id, "code": "print(1)", "language_id": 71}), content_type="application/json")
        self.assertEqual(Submission.objects.latest("id").language_id, 71)
        cache.clear()  # submit cooldown
        self.client.post("/api/submissions/", data=json.dumps(
            {"question": self.easy.id, "code": "select 1", "language_id": 82}), content_type="application/json")
        self.assertEqual(Submission.objects.latest("id").language_id, self.easy.language_id)


class QuizTimingTests(TestCase):
    def setUp(self):
        from .models import Quiz, QuizQuestion
        self.student = User.objects.create_user("qs", password="pw", role=User.Role.STUDENT, semester=1)
        self.course = Course.objects.create(name="C", slug="c-quiz")
        self.quiz = Quiz.objects.create(title="Q", course=self.course, duration_minutes=10, is_active=True)
        self.qq = QuizQuestion.objects.create(quiz=self.quiz, text="2+2", correct_option="B", points=5, order=1)
        self.client.force_login(self.student)

    def _age_attempt(self, minutes, seconds=0):
        from .models import QuizAttempt
        QuizAttempt.objects.filter(quiz=self.quiz).update(
            started_at=timezone.now() - timedelta(minutes=minutes, seconds=seconds))

    def test_page_shows_countdown(self):
        r = self.client.get(f"/quizzes/{self.quiz.id}/")
        self.assertContains(r, 'id="quiz-timer"')
        self.assertTrue(570 <= r.context["remaining_seconds"] <= 600)

    def test_answers_just_after_deadline_are_graded(self):
        self.client.get(f"/quizzes/{self.quiz.id}/")
        self._age_attempt(10, 20)  # auto-submit landed 20 s late
        self.client.post(f"/quizzes/{self.quiz.id}/", {f"q_{self.qq.id}": "B"})
        attempt = self.quiz.attempts.get()
        self.assertIsNotNone(attempt.finished_at)
        self.assertEqual(attempt.total_score, 5)

    def test_answers_long_after_deadline_are_refused_honestly(self):
        self.client.get(f"/quizzes/{self.quiz.id}/")
        self._age_attempt(15)
        r = self.client.post(f"/quizzes/{self.quiz.id}/", {f"q_{self.qq.id}": "B"}, follow=True)
        attempt = self.quiz.attempts.get()
        self.assertIsNone(attempt.finished_at)
        self.assertTrue(attempt.is_timed_out)
        self.assertContains(r, "time limit for this quiz had passed")

    def test_results_mid_quiz_go_back_to_the_quiz(self):
        self.client.get(f"/quizzes/{self.quiz.id}/")
        r = self.client.get(f"/quizzes/{self.quiz.id}/results/")
        self.assertRedirects(r, f"/quizzes/{self.quiz.id}/")

    def test_course_not_yet_available_for_semester(self):
        self.course.available_from_semester = 5
        self.course.save()
        r = self.client.get(f"/quizzes/{self.quiz.id}/")
        self.assertEqual(r.status_code, 302)
        self.assertFalse(self.quiz.attempts.exists())
