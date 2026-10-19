"""Certificate approval flow, question editing, DBMS modules and signup branches."""
import shutil
import tempfile
from io import StringIO
from unittest.mock import patch

from django.core.cache import cache
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import (
    Certificate,
    CertificateRequest,
    Course,
    Module,
    ModuleQuestionAssignment,
    Notification,
    Question,
    Submission,
    TestCase as QuestionTest,
    User,
)
from .services import certificate_eligible, overall_percentage

FAKE_PDF = b"%PDF-1.4 fake"


class MediaTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._media = tempfile.mkdtemp()
        cls._override = override_settings(MEDIA_ROOT=cls._media)
        cls._override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._override.disable()
        shutil.rmtree(cls._media, ignore_errors=True)
        super().tearDownClass()


def make_course(slug="c-programming", questions=5, category="c_programming", semester=1):
    course = Course.objects.create(name=slug.replace("-", " ").title(), slug=slug, available_from_semester=semester)
    module = Module.objects.create(name=f"{course.name} M1", course=course, category=category, order=1)
    qs = [
        Question.objects.create(module=module, title=f"Q{i}", slug=f"q{i}", description="d", is_mandatory=(i == 0))
        for i in range(questions)
    ]
    return course, module, qs


def solve(student, questions):
    for q in questions:
        Submission.objects.create(student=student, question=q, code="x", status=Submission.Status.ACCEPTED)


@patch("core.services.generate_certificate_pdf", return_value=FAKE_PDF)
class CertificateFlowTests(MediaTestCase):
    def setUp(self):
        cache.clear()
        self.course, self.module, self.questions = make_course()
        self.student = User.objects.create_user("NNM25CC001", password="pw", role=User.Role.STUDENT, usn="NNM25CC001")
        self.faculty = User.objects.create_user("fac", password="pw", role=User.Role.FACULTY, first_name="Asha")
        self.faculty.managed_courses.add(self.course)
        self.other_faculty = User.objects.create_user("fac2", password="pw", role=User.Role.FACULTY)
        self.hod = User.objects.create_user("hod", password="pw", role=User.Role.HOD)
        solve(self.student, self.questions)

    def apply(self):
        self.client.force_login(self.student)
        return self.client.get(reverse("certificate_create") + f"?course={self.course.pk}")

    def forward(self, req):
        self.client.force_login(self.faculty)
        return self.client.post(reverse("faculty_send_cert_request", args=[self.student.pk]),
                                {"request_id": req.pk, "notes": "verified"})

    def test_five_question_module_counts_fully(self, _pdf):
        self.assertEqual(overall_percentage(self.student, self.course), 100)
        self.assertTrue(certificate_eligible(self.student, self.course)[0])

    def test_full_flow_issues_one_certificate(self, pdf):
        self.assertEqual(self.apply().status_code, 200)
        req = CertificateRequest.objects.get(student=self.student, course=self.course)
        self.assertEqual(req.status, CertificateRequest.Status.PENDING_FACULTY)
        # only the faculty managing the course is notified
        self.assertTrue(Notification.objects.filter(recipient=self.faculty, related_student=self.student).exists())
        self.assertFalse(Notification.objects.filter(recipient=self.other_faculty).exists())
        self.apply()  # applying again doesn't duplicate the request
        self.assertEqual(CertificateRequest.objects.filter(student=self.student).count(), 1)

        self.client.force_login(self.faculty)
        page = self.client.get(reverse("faculty_cert_requests"))
        self.assertContains(page, f'name="request_id" value="{req.pk}"')
        self.assertContains(self.client.get(reverse("faculty_student_detail", args=[self.student.pk])), self.course.name)
        self.forward(req)
        req.refresh_from_db()
        self.assertEqual(req.status, CertificateRequest.Status.PENDING_HOD)
        self.client.force_login(self.hod)
        self.assertContains(self.client.get(reverse("hod_dashboard")), self.course.name)

        self.client.force_login(self.hod)
        self.client.post(reverse("role_select"), {"role": "hod"})
        r = self.client.post(reverse("hod_approve_certificate", args=[req.pk]), {"action": "approve"})
        self.assertRedirects(r, reverse("hod_dashboard"), fetch_redirect_response=False)
        cert = Certificate.objects.get(student=self.student, course=self.course)
        self.assertTrue(cert.pdf and cert.qr_code)
        self.assertEqual(pdf.call_args.kwargs["faculty"], self.faculty)  # the verifying faculty signs

        r = self.apply()
        self.assertRedirects(r, reverse("certificate_detail", args=[cert.pk]), fetch_redirect_response=False)
        page = self.client.get(reverse("certificate_detail", args=[cert.pk]))
        self.assertContains(page, self.course.name)
        download = self.client.get(reverse("certificate_download", args=[cert.pk]))
        self.assertEqual(b"".join(download.streaming_content), FAKE_PDF)
        self.assertEqual(self.client.get(reverse("certificate_verify", args=[cert.verification_hash])).status_code, 200)

        other = User.objects.create_user("NNM25CC002", password="pw", role=User.Role.STUDENT, usn="NNM25CC002")
        self.client.force_login(other)
        self.assertEqual(self.client.get(reverse("certificate_download", args=[cert.pk])).status_code, 403)

    def test_hod_decision_is_final_even_if_new_questions_were_added(self, _pdf):
        self.apply()
        req = CertificateRequest.objects.get(student=self.student)
        self.forward(req)
        Question.objects.create(module=self.module, title="New", slug="new", description="d", is_mandatory=True)
        self.assertFalse(certificate_eligible(self.student, self.course)[0])
        self.client.force_login(self.hod)
        self.client.get(reverse("hod_review_request", args=[req.pk]))  # opening it switches to the HoD role
        self.client.post(reverse("hod_approve_certificate", args=[req.pk]), {"action": "approve"})
        self.assertTrue(Certificate.objects.filter(student=self.student, course=self.course).exists())

    def test_hod_review_page_opens_from_faculty_mode(self, _pdf):
        self.apply()
        req = CertificateRequest.objects.get(student=self.student)
        self.forward(req)
        self.client.force_login(self.hod)
        session = self.client.session
        session["active_role"] = "faculty"
        session.save()
        r = self.client.get(reverse("hod_review_request", args=[req.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, self.course.name)

    def test_decline_needs_remarks(self, _pdf):
        self.apply()
        req = CertificateRequest.objects.get(student=self.student)
        self.forward(req)
        self.client.force_login(self.hod)
        self.client.get(reverse("hod_dashboard"))
        self.client.post(reverse("hod_approve_certificate", args=[req.pk]), {"action": "reject"})
        req.refresh_from_db()
        self.assertEqual(req.status, CertificateRequest.Status.PENDING_HOD)
        self.client.post(reverse("hod_approve_certificate", args=[req.pk]), {"action": "reject", "notes": "Redo Q2"})
        req.refresh_from_db()
        self.assertEqual(req.status, CertificateRequest.Status.REJECTED)
        self.assertFalse(Certificate.objects.exists())

    def test_faculty_cannot_forward_without_a_student_request(self, _pdf):
        self.client.force_login(self.faculty)
        self.client.post(reverse("faculty_send_cert_request", args=[self.student.pk]))
        self.assertFalse(CertificateRequest.objects.exists())

    def test_dashboard_offers_certificate_for_current_course(self, _pdf):
        python, _, py_qs = make_course("python-programming", category="python_programming", semester=3)
        self.student.semester = 3
        self.student.save()
        solve(self.student, py_qs)
        self.client.force_login(self.student)
        r = self.client.get(reverse("dashboard") + f"?course={python.pk}")
        self.assertTrue(r.context["certificate_eligible"])
        self.assertEqual(r.context["certificate_course"], python)

    def test_backfill_command_issues_missing_certificates(self, _pdf):
        CertificateRequest.objects.create(student=self.student, course=self.course, completion_percentage=90,
                                          status=CertificateRequest.Status.APPROVED, requested_by_faculty=self.faculty)
        call_command("generate_certificates", stdout=StringIO())
        cert = Certificate.objects.get(student=self.student, course=self.course)
        self.assertEqual(cert.completion_percentage, 90)
        call_command("generate_certificates", stdout=StringIO())
        self.assertEqual(Certificate.objects.count(), 1)


class CertificatePdfTests(MediaTestCase):
    def test_pdf_renders_with_special_characters_and_qr(self):
        from .certificate_generator import generate_certificate_pdf

        course = Course.objects.create(name="C & Data <Structures>", slug="c")
        student = User.objects.create_user("s", first_name="Anne & Co", usn="NNM1")
        pdf = generate_certificate_pdf(student, course, "19/10/2026", "http://localhost/verify/abc/")
        self.assertTrue(pdf.startswith(b"%PDF"))


class QuestionEditTests(TestCase):
    def setUp(self):
        self.course, self.module, (self.q,) = make_course(questions=1)
        self.q.csv_level = 3
        self.q.proctoring_enabled = True
        self.q.sample_input, self.q.sample_output = "1", "1"
        self.q.save()
        self.faculty = User.objects.create_user("fac", password="pw", role=User.Role.FACULTY)
        self.client.force_login(self.faculty)
        self.url = reverse("faculty_question_edit", args=[self.q.pk])

    def post_question(self, **overrides):
        data = {
            "module": self.module.pk, "title": "Renamed", "slug": "", "description": "desc", "difficulty": "easy",
            "csv_level": 3, "sample_input": "2", "sample_output": "4", "starter_code": "", "language_id": 50,
            "time_limit": 2.0, "memory_limit_kb": 128000, "proctoring_enabled": "on", "is_active": "on",
            "action": "save_question",
        }
        data.update(overrides)
        return self.client.post(self.url, data)

    def test_edit_saves_and_keeps_hidden_settings(self):
        r = self.post_question()
        self.assertRedirects(r, self.url, fetch_redirect_response=False)
        self.q.refresh_from_db()
        self.assertEqual(self.q.title, "Renamed")
        self.assertEqual(self.q.slug, "renamed")
        self.assertEqual(self.q.csv_level, 3)
        self.assertTrue(self.q.proctoring_enabled)
        sample = self.q.test_cases.get(is_sample=True)
        self.assertEqual((sample.stdin, sample.expected_output), ("2", "4"))

    def test_duplicate_sample_tests_do_not_crash(self):
        QuestionTest.objects.create(question=self.q, stdin="1", expected_output="1", is_sample=True, order=1)
        QuestionTest.objects.create(question=self.q, stdin="1", expected_output="1", is_sample=True, order=1)
        self.assertEqual(self.post_question().status_code, 302)

    def test_slug_clash_in_module_gets_suffix(self):
        Question.objects.create(module=self.module, title="Renamed", slug="renamed", description="d")
        self.post_question()
        self.q.refresh_from_db()
        self.assertEqual(self.q.slug, "renamed-1")

    def test_edit_and_delete_test_case(self):
        test = QuestionTest.objects.create(question=self.q, stdin="5", expected_output="25", order=2)
        r = self.client.post(self.url, {"action": "edit_test", "test_id": test.pk, "stdin": "6",
                                        "expected_output": "36", "order": 3})
        self.assertRedirects(r, self.url, fetch_redirect_response=False)
        test.refresh_from_db()
        self.assertEqual((test.stdin, test.expected_output, test.order, test.is_sample), ("6", "36", 3, False))
        self.client.post(self.url, {"action": "delete_test", "test_id": test.pk})
        self.assertFalse(QuestionTest.objects.filter(pk=test.pk).exists())

    def test_editing_first_sample_updates_question_sample(self):
        sample = QuestionTest.objects.create(question=self.q, stdin="1", expected_output="1", is_sample=True, order=1)
        self.client.post(self.url, {"action": "edit_test", "test_id": sample.pk, "stdin": "3",
                                    "expected_output": "9", "order": 1, "is_sample": "on"})
        self.q.refresh_from_db()
        self.assertEqual((self.q.sample_input, self.q.sample_output), ("3", "9"))

    def test_cannot_edit_another_questions_test(self):
        other = Question.objects.create(module=self.module, title="O", slug="o", description="d")
        test = QuestionTest.objects.create(question=other, stdin="1", expected_output="1")
        r = self.client.post(self.url, {"action": "delete_test", "test_id": test.pk})
        self.assertEqual(r.status_code, 404)


class DbmsModuleTests(TestCase):
    def setUp(self):
        self.course = Course.objects.create(name="DBMS", slug="dbms-sql", available_from_semester=1)
        self.module = Module.objects.create(name="Basics", course=self.course, category="dbms-sql", order=1)
        self.questions = [
            Question.objects.create(module=self.module, title=f"Q{i}", slug=f"q{i}", description="d",
                                    difficulty=Question.Difficulty.EASY, language_id=82)
            for i in range(5)
        ]
        self.student = User.objects.create_user("st", password="pw", role=User.Role.STUDENT)
        self.client.force_login(self.student)

    def test_module_opens_as_single_list(self):
        r = self.client.get(reverse("module_detail", args=[self.module.pk]))
        self.assertRedirects(r, reverse("module_level_detail", args=[self.module.pk, "all"]), fetch_redirect_response=False)
        r = self.client.get(reverse("module_level_detail", args=[self.module.pk, "medium"]))
        self.assertRedirects(r, reverse("module_level_detail", args=[self.module.pk, "all"]), fetch_redirect_response=False)
        r = self.client.get(reverse("module_level_detail", args=[self.module.pk, "all"]))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.context["questions"]), 5)
        self.assertEqual(ModuleQuestionAssignment.objects.filter(student=self.student).count(), 1)

    def test_question_page_keeps_single_assignment(self):
        self.client.get(reverse("module_level_detail", args=[self.module.pk, "all"]))
        self.client.get(reverse("question_detail", args=[self.questions[0].pk]))
        self.assertEqual(ModuleQuestionAssignment.objects.filter(student=self.student).count(), 1)

    def test_non_dbms_modules_reject_all(self):
        _, module, _ = make_course("c-programming")
        r = self.client.get(reverse("module_level_detail", args=[module.pk, "all"]))
        self.assertEqual(r.status_code, 403)


class SignupBranchTests(TestCase):
    def signup(self, branch):
        return self.client.post(reverse("signup"), {
            "username": "NNM25CC009", "first_name": "A", "last_name": "B", "email": "a@example.com",
            "department": branch, "semester": 3, "password1": "Str0ng-pass-123", "password2": "Str0ng-pass-123",
        })

    def test_branch_must_be_from_the_list(self):
        r = self.signup("My own branch")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(User.objects.filter(username="NNM25CC009").exists())

    def test_listed_branch_accepted(self):
        self.signup("CSE")
        user = User.objects.get(username="NNM25CC009")
        self.assertEqual((user.department, user.semester, user.year), ("CSE", 3, 2))
