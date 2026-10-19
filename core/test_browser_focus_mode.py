"""
T4 browser regression test: focus mode and "Read Problem" must never emit a
proctoring violation, and must never leave browser fullscreen.

Needs Playwright + Chromium, which are dev-only tools and not in
requirements.txt:

    pip install playwright && python -m playwright install chromium

Skipped automatically when Playwright isn't installed.
"""
import os
import unittest

from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import override_settings
from django.utils import timezone

from .models import AssignedQuestion, Module, ModuleQuestionAssignment, ProctoringEvent, Question, TestCase, User

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


@unittest.skipIf(sync_playwright is None, "Playwright not installed")
@override_settings(SESSION_COOKIE_SECURE=False, CSRF_COOKIE_SECURE=False, DEBUG=False)
class FocusModeProctoringBrowserTests(StaticLiveServerTestCase):
    @classmethod
    def setUpClass(cls):
        # Playwright's sync API runs an event loop on this thread; the ORM
        # calls in these tests are plain synchronous setup/assertions.
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
        self.student = User.objects.create_user("fm", password="pw", role=User.Role.STUDENT)
        self.module = Module.objects.create(name="Focus")

    def _question(self, proctored):
        q = Question.objects.create(
            module=self.module,
            title=f"Q proctored={proctored}",
            slug=f"q-{proctored}",
            description="Read two integers and print their sum.",
            proctoring_enabled=proctored,
            starter_code="#include <stdio.h>\nint main(){return 0;}\n",
        )
        TestCase.objects.create(question=q, stdin="1 2", expected_output="3", is_sample=True)
        a, _ = ModuleQuestionAssignment.objects.get_or_create(student=self.student, module=self.module, difficulty="easy")
        AssignedQuestion.objects.create(assignment=a, question=q, order=q.pk, unlocked_at=timezone.now())
        return q

    def _open(self, question):
        self.client.force_login(self.student)
        context = self.browser.new_context(viewport={"width": 1280, "height": 800})
        context.add_cookies([{
            "name": "sessionid",
            "value": self.client.cookies["sessionid"].value,
            "url": self.live_server_url,
        }])
        page = context.new_page()
        calls = {"violation": [], "audit": []}
        page.on("request", lambda r: calls["violation"].append(r.post_data) if "/api/report-violation/" in r.url else None)
        page.on("request", lambda r: calls["audit"].append(r.post_data) if "/api/proctoring/audit/" in r.url else None)
        page.goto(f"{self.live_server_url}/questions/{question.id}/")
        page.wait_for_selector(".monaco-editor", timeout=20000)
        return context, page, calls

    def _cycle_focus_mode(self, page):
        body = page.locator("body")
        page.locator(".monaco-editor .view-lines").click()  # clicking the editor enters focus mode
        self.assertIn("is-maximized-page", body.get_attribute("class"))

        page.click("#editor-toggle-desc-btn")  # Read Problem
        self.assertIn("is-maximized-split", body.get_attribute("class"))
        self.assertTrue(page.locator("#problem-statement").is_visible())
        page.click("#editor-toggle-desc-btn")  # Hide Problem: back to the editor, still in focus mode
        self.assertNotIn("is-maximized-split", body.get_attribute("class"))
        self.assertIn("is-maximized-page", body.get_attribute("class"))

        page.keyboard.press("Control+Backslash")
        self.assertIn("is-maximized-split", body.get_attribute("class"))
        page.keyboard.press("Control+Backslash")
        self.assertNotIn("is-maximized-split", body.get_attribute("class"))

        page.click("#editor-minimize-btn")  # Exit focus mode
        self.assertNotIn("is-maximized-page", body.get_attribute("class"))

        # Read Problem open -> Exit focus mode in one step.
        page.locator(".monaco-editor .view-lines").click()
        page.click("#editor-toggle-desc-btn")
        page.click("#editor-minimize-btn")
        cls = body.get_attribute("class")
        self.assertNotIn("is-maximized-page", cls)
        self.assertNotIn("is-maximized-split", cls)
        page.wait_for_timeout(1000)  # let blur/visibility debounce timers (350ms) elapse

    def test_focus_mode_emits_no_violation_and_keeps_fullscreen_when_proctored(self):
        q = self._question(proctored=True)
        context, page, calls = self._open(q)
        try:
            page.click("#take-me-to-question-btn")  # starts proctoring; requests fullscreen
            page.wait_for_function("document.fullscreenElement !== null", timeout=5000)

            self._cycle_focus_mode(page)

            self.assertEqual(calls["violation"], [], "focus-mode transitions must not report violations")
            self.assertTrue(page.evaluate("document.fullscreenElement !== null"), "focus mode must not leave fullscreen")
            self.assertEqual(ProctoringEvent.objects.filter(kind="violation").count(), 0)
            # Audit trail recorded the transitions (not as violations).
            kinds = set(ProctoringEvent.objects.values_list("kind", flat=True))
            self.assertTrue({"focus_enter", "focus_exit", "problem_open", "problem_close"} <= kinds, kinds)

            # Control: genuinely leaving fullscreen IS still a violation, so the
            # zero above isn't because the listeners are broken.
            page.evaluate("document.exitFullscreen()")
            page.wait_for_function("document.fullscreenElement === null", timeout=5000)
            page.wait_for_timeout(500)
            self.assertEqual(len(calls["violation"]), 1)
            self.assertIn("Exited fullscreen mode", calls["violation"][0])
        finally:
            context.close()

    def test_focus_mode_never_touches_browser_fullscreen_when_not_proctored(self):
        q = self._question(proctored=False)
        context, page, calls = self._open(q)
        try:
            self._cycle_focus_mode(page)
            self.assertTrue(page.evaluate("document.fullscreenElement === null"))
            self.assertEqual(calls["violation"], [])
            self.assertEqual(calls["audit"], [])  # audit only runs under proctoring
        finally:
            context.close()

    def test_escape_closes_problem_then_focus_mode(self):
        q = self._question(proctored=False)
        context, page, calls = self._open(q)
        try:
            body = page.locator("body")
            page.locator(".monaco-editor .view-lines").click()
            page.click("#editor-toggle-desc-btn")
            page.keyboard.press("Escape")
            self.assertNotIn("is-maximized-split", body.get_attribute("class"))
            self.assertIn("is-maximized-page", body.get_attribute("class"))
            page.keyboard.press("Escape")
            self.assertNotIn("is-maximized-page", body.get_attribute("class"))
        finally:
            context.close()

    def test_results_reachable_in_focus_mode_and_dashboard_leaves_without_violation(self):
        q = self._question(proctored=True)
        context, page, calls = self._open(q)
        tests = '[{"passed": false, "stdin": "1 2", "expected": "3", "actual": "0"}]'
        page.route("**/api/run/", lambda route: route.fulfill(status=200, content_type="application/json",
                                                               body='{"tests": %s}' % tests))
        try:
            page.click("#take-me-to-question-btn")
            page.wait_for_function("document.fullscreenElement !== null", timeout=5000)
            page.locator(".monaco-editor .view-lines").click()  # focus mode
            self.assertTrue(page.locator("#editor-dashboard-link").is_visible())

            # Results of a run/submit scroll into view instead of rendering off-screen.
            page.click("[data-run-button]")
            page.wait_for_selector("#test-results", state="visible")
            page.wait_for_timeout(1000)  # smooth scroll
            top = page.evaluate("document.getElementById('test-results').getBoundingClientRect().top")
            self.assertLess(top, 200)
            self.assertEqual(page.evaluate("document.activeElement.id"), "results-title")

            page.click("#results-back-to-editor")
            page.wait_for_timeout(1000)
            self.assertLess(page.evaluate("document.getElementById('editor-chrome-container').getBoundingClientRect().top"), 200)

            # The mouse wheel over the editor reaches the results again.
            page.mouse.move(640, 400)
            page.mouse.wheel(0, 3000)
            page.wait_for_timeout(800)
            self.assertGreater(page.evaluate("document.querySelector('.submit-panel-wide').scrollTop"), 300)

            page.locator(".results-actions a[data-leave-question]").click()
            page.wait_for_url(f"{self.live_server_url}/", timeout=10000)
            page.wait_for_timeout(500)
            self.assertEqual(calls["violation"], [], "leaving via Dashboard must not report a violation")
            self.assertEqual(ProctoringEvent.objects.filter(kind="violation").count(), 0)
        finally:
            context.close()
