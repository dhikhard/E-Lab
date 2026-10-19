"""T4/F12: proctoring events are stored server-side and focus-mode audit events are never violations."""
import json
from unittest.mock import patch

from django.core.cache import cache
from django.test import TestCase
from django.utils import timezone

from .models import AssignedQuestion, Module, ModuleQuestionAssignment, ProctoringEvent, Question, Submission, User


class ProctoringEventTests(TestCase):
    def setUp(self):
        cache.clear()  # rate limits are keyed by user id, which the test DB reuses between tests
        self.student = User.objects.create_user("p1", password="pw", role=User.Role.STUDENT)
        self.module = Module.objects.create(name="M")
        self.q = Question.objects.create(
            module=self.module, title="Q", slug="q", description="d", proctoring_enabled=True
        )
        a = ModuleQuestionAssignment.objects.create(student=self.student, module=self.module, difficulty="easy")
        AssignedQuestion.objects.create(assignment=a, question=self.q, order=1, unlocked_at=timezone.now())
        self.client.force_login(self.student)

    def report(self, reason="Tab switched or minimized", question_id=None):
        return self.client.post(
            "/api/report-violation/",
            data=json.dumps({"question_id": question_id or self.q.id, "reason": reason}),
            content_type="application/json",
        )

    def audit(self, event):
        return self.client.post(
            "/api/proctoring/audit/",
            data=json.dumps({"question_id": self.q.id, "event": event}),
            content_type="application/json",
        )

    def test_violation_persisted_in_db(self):
        self.assertEqual(self.report().status_code, 200)
        ev = ProctoringEvent.objects.get()
        self.assertEqual(ev.kind, ProctoringEvent.Kind.VIOLATION)
        self.assertEqual(ev.reason, "Tab switched or minimized")

    def test_violations_survive_logout(self):
        self.report()
        self.client.logout()
        self.client.force_login(self.student)
        self.assertEqual(ProctoringEvent.objects.filter(student=self.student, kind="violation").count(), 1)

    def test_reason_is_truncated(self):
        self.report(reason="x" * 5000)
        self.assertLessEqual(len(ProctoringEvent.objects.get().reason), 200)

    def test_unknown_question_rejected(self):
        self.assertEqual(self.report(question_id=999999).status_code, 400)
        self.assertEqual(self.client.post("/api/report-violation/", data="{}", content_type="application/json").status_code, 400)

    def test_report_rate_limited(self):
        codes = [self.report().status_code for _ in range(40)]
        self.assertIn(429, codes)
        self.assertLessEqual(ProctoringEvent.objects.count(), 30)

    def test_focus_mode_audit_events_are_not_violations(self):
        for event in ("focus_enter", "problem_open", "problem_close", "focus_exit"):
            self.assertEqual(self.audit(event).status_code, 200)
        self.assertEqual(ProctoringEvent.objects.filter(kind=ProctoringEvent.Kind.VIOLATION).count(), 0)
        self.assertEqual(ProctoringEvent.objects.count(), 4)

    def test_audit_rejects_unknown_event(self):
        self.assertEqual(self.audit("violation").status_code, 400)
        self.assertEqual(self.audit("anything").status_code, 400)
        self.assertEqual(ProctoringEvent.objects.count(), 0)

    @patch("core.views.evaluate_submission_task.delay")
    def test_submission_claims_unclaimed_violations(self, _delay):
        self.report("Tab switched or minimized")
        self.report("Exited fullscreen mode")
        self.audit("focus_enter")
        r = self.client.post(
            "/api/submissions/", data=json.dumps({"question": self.q.id, "code": "x"}), content_type="application/json"
        )
        self.assertEqual(r.status_code, 201)
        sub = Submission.objects.get()
        self.assertEqual(sub.proctoring_violations, 2)
        self.assertEqual(len(sub.proctoring_logs), 2)
        self.assertIn("Exited fullscreen mode", sub.proctoring_logs[1])
        self.assertEqual(ProctoringEvent.objects.filter(submission=sub).count(), 3)

        # Next submission starts from zero.
        from django.core.cache import cache
        cache.clear()
        self.client.post(
            "/api/submissions/", data=json.dumps({"question": self.q.id, "code": "y"}), content_type="application/json"
        )
        self.assertEqual(Submission.objects.order_by("-id").first().proctoring_violations, 0)

    @patch("core.views.evaluate_submission_task.delay")
    def test_form_submission_claims_violations(self, _delay):
        self.report()
        self.client.post(f"/questions/{self.q.id}/", data={"code": "int main(){}"})
        sub = Submission.objects.get()
        self.assertEqual(sub.proctoring_violations, 1)
