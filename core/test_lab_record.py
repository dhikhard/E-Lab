"""T1: printable lab record — course title, running footer, no N+1 queries, PDF evidence."""
import datetime
import io
import os
import unittest

from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.db import connection
from django.test import TestCase, override_settings
from django.test.utils import CaptureQueriesContext

from .views import _usn_watermark_url
from .models import Course, Module, Question, Submission, TestCase as QTestCase, User

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

FOOTER_LEFT = "NMAM Institute of Technology"
FOOTER_RIGHT = "Department of Computer and Communication"

# 1x1 transparent PNG
PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def make_record(student, course_name="Python Programming", modules=1, questions=1, code="print(1)", description="d"):
    course = Course.objects.create(name=course_name, slug=course_name.lower().replace(" ", "-"))
    for m in range(modules):
        module = Module.objects.create(name=f"{course_name} M{m}", course=course, order=m + 1)
        for i in range(questions):
            q = Question.objects.create(
                module=module, title=f"{course_name} Q{m}.{i}", slug=f"{course.slug}-{m}-{i}", description=description
            )
            QTestCase.objects.create(question=q, stdin="1", expected_output="1", is_sample=True)
            Submission.objects.create(student=student, question=q, code=code, status="accepted", score=100)
    return course


class LabRecordViewTests(TestCase):
    def setUp(self):
        self.student = User.objects.create_user("lr", password="pw", role=User.Role.STUDENT, semester=8)
        self.client.force_login(self.student)

    def test_title_uses_course_name(self):
        make_record(self.student, "Python Programming")
        r = self.client.get("/lab-record/")
        self.assertEqual(r.context["record_title"], "Python Programming Laboratory Record")
        self.assertNotContains(r, "C Programming Laboratory Record")

    def test_mixed_courses_get_generic_title(self):
        make_record(self.student, "Python Programming")
        make_record(self.student, "Java Programming")
        r = self.client.get("/lab-record/")
        self.assertEqual(r.context["record_title"], "Programming Laboratory Record")
        self.assertTrue(r.context["show_course"])

    def test_footer_text_present(self):
        make_record(self.student)
        r = self.client.get("/lab-record/")
        self.assertContains(r, FOOTER_LEFT)
        self.assertContains(r, FOOTER_RIGHT)
        self.assertNotContains(r, "Commnunication")

    def test_cover_page_uses_block_svg(self):
        make_record(self.student)
        r = self.client.get("/lab-record/")
        self.assertContains(r, "img/BLOCK.svg")

    def test_index_lists_completed_programs_with_submission_date(self):
        make_record(self.student, "Python Programming", modules=2, questions=2)
        sub = Submission.objects.filter(student=self.student).order_by("id").first()
        Submission.objects.filter(pk=sub.pk).update(submitted_at=datetime.datetime(2025, 8, 26, 10, tzinfo=datetime.timezone.utc))
        # An unsolved question must not be listed.
        Question.objects.create(module=sub.question.module, title="Unsolved", slug="unsolved", description="d")
        r = self.client.get("/lab-record/")
        rows = r.context["index_rows"]
        self.assertEqual([row["number"] for row in rows], [1, 2, 3, 4])
        self.assertEqual(rows[0]["question"], sub.question)
        self.assertContains(r, "26/08/2025")
        self.assertNotContains(r, "Unsolved")

    def test_watermark_is_the_students_usn(self):
        self.student.usn = "4NM22CC001"
        self.student.save()
        make_record(self.student)
        r = self.client.get("/lab-record/")
        self.assertIn("4NM22CC001", r.context["watermark_url"])
        self.assertTrue(r.context["watermark_url"].startswith("data:image/svg+xml,"))

    def test_watermark_escapes_markup(self):
        url = _usn_watermark_url("<a'\")")
        self.assertNotIn("<a", url)
        self.assertNotIn('"', url)
        self.assertNotIn("'", url)
        self.assertNotIn(")", url)

    def test_query_count_does_not_grow_with_questions(self):
        make_record(self.student, "Small", modules=1, questions=1)
        with CaptureQueriesContext(connection) as small:
            self.client.get("/lab-record/")
        make_record(self.student, "Big", modules=3, questions=6)
        with CaptureQueriesContext(connection) as big:
            self.client.get("/lab-record/")
        self.assertLessEqual(len(big), len(small) + 2)


@unittest.skipIf(sync_playwright is None, "Playwright not installed")
@override_settings(SESSION_COOKIE_SECURE=False, CSRF_COOKIE_SECURE=False)
class LabRecordPdfTests(StaticLiveServerTestCase):
    """Prints the lab record to PDF with Chromium and checks every page's footer."""

    @classmethod
    def setUpClass(cls):
        os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
        super().setUpClass()
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        super().tearDownClass()

    def setUp(self):
        self.student = User.objects.create_user("pdf", password="pw", role=User.Role.STUDENT, semester=8)

    def _pdf(self, name):
        from pypdf import PdfReader

        self.client.force_login(self.student)
        context = self.browser.new_context()
        context.add_cookies([{"name": "sessionid", "value": self.client.cookies["sessionid"].value, "url": self.live_server_url}])
        page = context.new_page()
        page.goto(f"{self.live_server_url}/lab-record/")
        screen_footer_visible = page.locator(".print-footer").is_visible()
        pdf = page.pdf(format="A4")
        context.close()
        out_dir = os.environ.get("LAB_RECORD_PDF_DIR")
        if out_dir:
            with open(os.path.join(out_dir, f"{name}.pdf"), "wb") as fh:
                fh.write(pdf)
        reader = PdfReader(io.BytesIO(pdf))
        return [p.extract_text() for p in reader.pages], screen_footer_visible

    def _assert_footer_on_every_page(self, pages):
        self.assertNotIn(FOOTER_LEFT, pages[0], "cover page has no footer")
        for number, text in enumerate(pages[1:], start=2):
            with self.subTest(page=number):
                self.assertIn(FOOTER_LEFT, text)
                self.assertIn(FOOTER_RIGHT, text)
                self.assertRegex(text, rf"(^|\s){number}(\s|$)")

    def test_one_page_report(self):
        make_record(self.student)
        pages, screen_visible = self._pdf("one_page")
        self.assertEqual(len(pages), 3)  # cover, index, content
        self.assertIn("INDEX", pages[1])
        self._assert_footer_on_every_page(pages)
        self.assertFalse(screen_visible, "footer must not appear on screen")

    def test_multi_page_report(self):
        make_record(self.student, modules=3, questions=3)
        pages, _ = self._pdf("multi_page")
        self.assertGreaterEqual(len(pages), 3)
        self._assert_footer_on_every_page(pages)

    def test_unbreakable_code_block_longer_than_a_page(self):
        long_code = "\n".join(f"line_{i} = {i}  # " + "x" * 60 for i in range(150)) + "\n" + "y" * 400
        make_record(self.student, code=long_code)
        pages, _ = self._pdf("long_code")
        self.assertGreaterEqual(len(pages), 2)
        self._assert_footer_on_every_page(pages)
        self.assertIn("line_149", "".join(pages))  # nothing lost under the footer

    def test_report_with_image(self):
        make_record(self.student, description=f'<p>See figure:</p><img src="{PNG}" width="600" height="400" alt="fig">')
        pages, _ = self._pdf("image")
        self._assert_footer_on_every_page(pages)
