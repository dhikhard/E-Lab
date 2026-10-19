from unittest.mock import patch

from django.http import HttpResponse
from django.test import TestCase
from django.urls import reverse

from .models import User


class DashboardRoutingTests(TestCase):
    def test_admin_is_sent_to_admin_dashboard(self):
        admin = User.objects.create_superuser(
            username="admin",
            email="admin@example.com",
            password="password123",
            role=User.Role.ADMIN,
        )
        self.client.force_login(admin)

        response = self.client.get(reverse("dashboard"))

        self.assertRedirects(response, reverse("admin:index"), fetch_redirect_response=False)

    def test_faculty_still_sees_faculty_dashboard(self):
        faculty = User.objects.create_user(
            username="faculty",
            password="password123",
            role=User.Role.FACULTY,
        )
        self.client.force_login(faculty)

        with patch("core.views.render", return_value=HttpResponse()) as render_mock:
            response = self.client.get(reverse("dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(render_mock.call_args.args[1], "faculty/dashboard.html")


class HintSystemTests(TestCase):
    def setUp(self):
        from django.core.cache import cache

        from .models import Course, Module, Question

        cache.clear()  # per-student unlock rate limits are keyed by user id, which the test DB reuses
        self.student = User.objects.create_user(
            username="test_student",
            password="password123",
            role=User.Role.STUDENT,
        )
        self.course = Course.objects.create(name="C Programming", slug="c-programming")
        self.module = Module.objects.create(name="Module 1", course=self.course, order=1)
        self.question = Question.objects.create(
            title="Sum of Two Numbers",
            slug="sum-of-two",
            module=self.module,
            description="Given two integers, print their sum.",
            difficulty=Question.Difficulty.EASY,
        )

    @patch("core.hint_service.enqueue_hint_generation")
    def test_progressive_hints_capped_at_three(self, _mock_enqueue):
        from .hint_service import generate_hint_for_submission, get_student_hints_for_question
        from .models import StudentQuestionHint, Submission

        # 1st unsuccessful submission
        sub1 = Submission.objects.create(
            student=self.student,
            question=self.question,
            code="int main() { return 0; }",
            status=Submission.Status.WRONG_ANSWER,
            judge_output='[{"passed": false, "stdin": "2 3", "expected": "5", "actual": "0"}]',
        )
        h1 = generate_hint_for_submission(sub1)
        self.assertIsNotNone(h1)
        self.assertEqual(h1.hint_number, 1)
        self.assertTrue(len(h1.hint_text) > 10)

        # 2nd unsuccessful submission
        sub2 = Submission.objects.create(
            student=self.student,
            question=self.question,
            code="int main() { while(1); }",
            status=Submission.Status.TLE,
            error_output="Time Limit Exceeded",
        )
        h2 = generate_hint_for_submission(sub2)
        self.assertIsNotNone(h2)
        self.assertEqual(h2.hint_number, 2)
        self.assertNotEqual(h1.hint_text, h2.hint_text)

        # 3rd unsuccessful submission
        sub3 = Submission.objects.create(
            student=self.student,
            question=self.question,
            code="int main() { char *p = NULL; *p = 1; }",
            status=Submission.Status.RUNTIME_ERROR,
            error_output="Segmentation fault",
        )
        h3 = generate_hint_for_submission(sub3)
        self.assertIsNotNone(h3)
        self.assertEqual(h3.hint_number, 3)

        # 4th unsuccessful submission -> MUST NOT create a 4th hint
        sub4 = Submission.objects.create(
            student=self.student,
            question=self.question,
            code="int main() { return 1; }",
            status=Submission.Status.WRONG_ANSWER,
        )
        h4 = generate_hint_for_submission(sub4)
        total_hints = StudentQuestionHint.objects.filter(student=self.student, question=self.question).count()
        self.assertEqual(total_hints, 3)

        hints_info = get_student_hints_for_question(self.student, self.question)
        self.assertEqual(hints_info["unlocked_count"], 3)
        self.assertFalse(hints_info["can_unlock_more"])

    def test_submission_hints_api(self):
        from .models import Submission, StudentQuestionHint

        sub = Submission.objects.create(
            student=self.student,
            question=self.question,
            code="int main() { return 0; }",
            status=Submission.Status.WRONG_ANSWER,
        )
        StudentQuestionHint.objects.create(
            student=self.student,
            question=self.question,
            submission=sub,
            hint_number=1,
            hint_text="Check boundary constraints.",
            hint_type="diagnostic",
        )

        self.client.force_login(self.student)
        response = self.client.get(reverse("submission_hints_api", args=[sub.id]))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["unlocked_count"], 1)
        self.assertEqual(data["hints"][0]["hint_number"], 1)
        self.assertEqual(data["hints"][0]["hint_text"], "Check boundary constraints.")

    @patch("core.hint_service.enqueue_hint_generation")
    def test_unlock_question_hint_api(self, _mock_enqueue):
        from django.utils import timezone
        from .models import AssignedQuestion, ModuleQuestionAssignment
        # Hint unlock is gated to questions assigned to the student (audit F16).
        assignment = ModuleQuestionAssignment.objects.create(
            student=self.student, module=self.module, difficulty=self.question.difficulty
        )
        AssignedQuestion.objects.create(
            assignment=assignment, question=self.question, order=1, unlocked_at=timezone.now()
        )
        self.client.force_login(self.student)

        # Unlock Hint 1
        res1 = self.client.post(reverse("unlock_question_hint_api", args=[self.question.id]))
        self.assertEqual(res1.status_code, 200)
        d1 = res1.json()
        self.assertTrue(d1["success"])
        self.assertEqual(d1["unlocked_count"], 1)
        self.assertEqual(d1["unlocked_hint"]["hint_number"], 1)
        self.assertTrue(d1["can_unlock_more"])

        # Unlock Hint 2
        res2 = self.client.post(reverse("unlock_question_hint_api", args=[self.question.id]))
        self.assertEqual(res2.status_code, 200)
        d2 = res2.json()
        self.assertEqual(d2["unlocked_count"], 2)
        self.assertEqual(d2["unlocked_hint"]["hint_number"], 2)

        # Unlock Hint 3
        res3 = self.client.post(reverse("unlock_question_hint_api", args=[self.question.id]))
        self.assertEqual(res3.status_code, 200)
        d3 = res3.json()
        self.assertEqual(d3["unlocked_count"], 3)
        self.assertFalse(d3["can_unlock_more"])

        # Attempt 4th unlock -> should return 400
        res4 = self.client.post(reverse("unlock_question_hint_api", args=[self.question.id]))
        self.assertEqual(res4.status_code, 400)
        d4 = res4.json()
        self.assertFalse(d4["success"])



from django.core.cache import cache
from django.test import override_settings
from unittest.mock import patch, MagicMock

LOCMEM_CACHE = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}


@override_settings(CACHES=LOCMEM_CACHE)
class LlmHintTaskTests(TestCase):
    def setUp(self):
        from .models import Course, Module, Question, StudentQuestionHint, Submission
        cache.clear()
        self.student = User.objects.create_user(
            username="hint_student", password="password123", role=User.Role.STUDENT,
        )
        self.course = Course.objects.create(name="C Programming", slug="c-programming-llm")
        self.module = Module.objects.create(name="Module 1", course=self.course, order=1)
        self.question = Question.objects.create(
            title="Two Sum", slug="two-sum-llm", module=self.module,
            description="Given an array and target, find two numbers that add up to target.",
            difficulty=Question.Difficulty.EASY,
        )
        self.submission = Submission.objects.create(
            student=self.student, question=self.question, code="pass",
            status=Submission.Status.WRONG_ANSWER,
            judge_output='[{"passed": false, "stdin": "4 9", "expected": "2", "actual": "3"}]',
        )
        StudentQuestionHint.objects.create(
            student=self.student, question=self.question, submission=self.submission,
            hint_number=1, hint_text="old diagnostic text", hint_type="diagnostic",
        )

    # Updated for the T3 hint engine: one task now generates all three tiers per
    # (question, language) instead of one tier per call. Same guarantees asserted:
    # the LLM result is cached, previously-shown fallback rows are upgraded, and
    # a model failure doesn't crash or poison the cache.
    LLM_JSON = (
        '{"hints": [{"level": 1, "text": "Think about what you need to remember about numbers you have already seen."},'
        ' {"level": 2, "text": "A structure with fast lookup by value would let you avoid a second scan."},'
        ' {"level": 3, "text": "Check each number against what is still needed before storing it, so you never pair it with itself."}]}'
    )

    def test_llm_task_caches_llm_text_and_upgrades_rows(self):
        from . import llm_hints
        from .models import GeneratedHintSet, StudentQuestionHint
        from .tasks import generate_hint_set_task

        StudentQuestionHint.objects.filter(student=self.student, question=self.question).update(language="c")
        with patch("core.llm_hints._call_model", return_value=(self.LLM_JSON, {"eval_count": 80})):
            result = generate_hint_set_task.apply(args=[self.question.id, "c"])

        self.assertEqual(len(result.result), 3)
        self.assertEqual(cache.get(llm_hints.cache_key(self.question, "c"))[1], result.result[1])
        self.assertTrue(GeneratedHintSet.objects.filter(question=self.question, language="c").exists())
        row = StudentQuestionHint.objects.get(student=self.student, question=self.question, hint_number=1)
        self.assertEqual(row.hint_type, "local_llm")
        self.assertIn("already seen", row.hint_text)

    def test_llm_task_falls_back_without_crashing(self):
        import requests
        from . import llm_hints
        from .hint_service import generation_status
        from .tasks import generate_hint_set_task

        with patch("core.llm_hints._call_model", side_effect=requests.Timeout("slow")):
            result = generate_hint_set_task.apply(args=[self.question.id, "c"])

        self.assertIsNone(result.result)
        self.assertIsNone(cache.get(llm_hints.cache_key(self.question, "c")))
        self.assertEqual(generation_status(self.question, "c"), "unavailable")


@override_settings(CACHES=LOCMEM_CACHE)
class PregenerateCacheTests(TestCase):
    def setUp(self):
        from .models import Course, Module, Question
        cache.clear()
        self.course = Course.objects.create(name="C++", slug="cpp-pregen")
        self.module = Module.objects.create(name="M", course=self.course, order=1)
        self.question = Question.objects.create(
            title="Reverse", slug="reverse-pregen", module=self.module,
            description="Reverse a string.", difficulty=Question.Difficulty.EASY,
            allow_multiple_languages=True,
        )

    def test_pregenerate_does_not_poison_llm_cache_slot(self):
        from . import llm_hints
        from .hint_service import pregenerate_hints_for_question
        with patch("core.tasks.generate_hint_set_task.apply_async") as apply_async:
            results = pregenerate_hints_for_question(self.question)
            again = pregenerate_hints_for_question(self.question)

        # One job per student language; a second call while in flight queues nothing.
        self.assertEqual(apply_async.call_count, 4)
        self.assertEqual(set(results), {"c", "cpp", "python", "java"})
        self.assertTrue(all(state == "in-progress" for state in again.values()))
        for language in results:
            self.assertIsNone(cache.get(llm_hints.cache_key(self.question, language)))


class DiagnosticHintGroundingTests(TestCase):
    def setUp(self):
        from .models import Course, Module, Question
        self.student = User.objects.create_user(
            username="diag_student", password="password123", role=User.Role.STUDENT,
        )
        self.course = Course.objects.create(name="Python", slug="python-diag")
        self.module = Module.objects.create(name="Strings", course=self.course, order=1)
        self.question = Question.objects.create(
            title="Palindrome Check", slug="palindrome-diag", module=self.module,
            description="Check if a string is a palindrome.", difficulty=Question.Difficulty.EASY,
        )

    def test_wrong_answer_hint_references_failing_test(self):
        from .hint_service import _generate_diagnostic_hint_for_question
        from .models import Submission
        sub = Submission.objects.create(
            student=self.student, question=self.question, code="pass",
            status=Submission.Status.WRONG_ANSWER,
            judge_output='[{"passed": false, "stdin": "abba", "expected": "yes", "actual": "no"}]',
        )
        hint = _generate_diagnostic_hint_for_question(self.question, 1, sub)
        self.assertIn("abba", hint)
        self.assertIn("Your first failing test", hint)


class RagAgentStarterCodeTests(TestCase):
    def test_starter_code_is_python_not_c(self):
        from .rag_agent import RAGQuestionAgent
        agent = RAGQuestionAgent()
        starter = agent._adapt_starter_code("", "easy")
        self.assertIn("def solve", starter)
        self.assertNotIn("#include", starter)

    def test_clean_tc_value_is_callable_on_instance(self):
        from .rag_agent import RAGQuestionAgent
        agent = RAGQuestionAgent()
        self.assertEqual(agent._clean_tc_value("```\n5\n```"), "5")


class DbmsProgressTests(TestCase):
    def test_progress_can_reach_100_percent(self):
        from .models import Course, Module, Question, SubQuestion, Submission
        from .services import update_progress
        student = User.objects.create_user(username="dbms_st", password="password123", role=User.Role.STUDENT)
        course = Course.objects.create(name="DBMS", slug="dbms-prog")
        module = Module.objects.create(name="SQL", course=course, order=1, category="dbms_module")
        q = Question.objects.create(title="Q", slug="dbms-q", module=module, description="d", difficulty=Question.Difficulty.EASY)
        for i, t in enumerate(["select", "insert", "update"]):
            sq = SubQuestion.objects.create(main_question=q, type=t, title=t, slug=t, description=t)
            Submission.objects.create(student=student, question=q, subquestion=sq, code="x", status=Submission.Status.ACCEPTED)
        update_progress(student, module)
        from .models import Progress
        p = Progress.objects.get(student=student, module=module)
        self.assertEqual(p.percentage, 100)
