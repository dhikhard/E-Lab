"""F9: stored-XSS protection for authored descriptions."""
from django.template import Context, Template
from django.test import SimpleTestCase

from .html_sanitizer import sanitize_html


class SanitizerTests(SimpleTestCase):
    def test_keeps_basic_formatting(self):
        html = "<p>Read <strong>n</strong> and <em>n</em> ints.</p><ul><li>a</li></ul><pre><code>1 2</code></pre>"
        self.assertEqual(sanitize_html(html), html)

    def test_plain_text_and_comparisons_survive(self):
        self.assertEqual(sanitize_html("1 <= n <= 10^5 & a < b"), "1 &lt;= n &lt;= 10^5 &amp; a &lt; b")

    def test_strips_scripts_and_handlers(self):
        attacks = [
            '<script>alert(1)</script>ok',
            '<img src=x onerror="alert(1)">',
            '<a href="javascript:alert(1)">x</a>',
            '<a href=" JaVaScRiPt:alert(1)">x</a>',
            '<svg onload=alert(1)><circle/></svg>',
            '<iframe src="https://evil"></iframe>',
            '<p style="background:url(javascript:alert(1))" onclick="x()">t</p>',
            '<img src="data:text/html;base64,PHNjcmlwdD4=">',
            '<scr<script>ipt>alert(1)</script>',
        ]
        for attack in attacks:
            with self.subTest(attack=attack):
                out = sanitize_html(attack).lower()
                self.assertNotIn("<script", out)
                self.assertNotIn("onerror", out)
                self.assertNotIn("onload", out)
                self.assertNotIn("onclick", out)
                self.assertNotIn("javascript:", out)
                self.assertNotIn("<iframe", out)
                self.assertNotIn("data:text", out)
                self.assertNotIn("style=", out)

    def test_safe_links_and_images_kept(self):
        out = sanitize_html('<a href="https://nitte.edu.in">site</a><img src="data:image/png;base64,AAA" alt="fig">')
        self.assertIn('href="https://nitte.edu.in"', out)
        self.assertIn('rel="noopener noreferrer"', out)
        self.assertIn('src="data:image/png;base64,AAA"', out)

    def test_unclosed_tags_are_closed(self):
        self.assertEqual(sanitize_html("<p><strong>bold"), "<p><strong>bold</strong></p>")

    def test_template_filter(self):
        t = Template("{% load custom_filters %}{{ d|safe_html|linebreaks }}")
        out = t.render(Context({"d": "<b>x</b><script>bad()</script>\nline2"}))
        self.assertIn("<b>x</b>", out)
        self.assertNotIn("bad()", out)
        self.assertIn("line2", out)
