"""
Allowlist HTML sanitiser for question/quiz text authored by faculty, imports
or the RAG generator (audit F9). Standard library only.

Keeps basic formatting (paragraphs, emphasis, lists, code, tables, images,
links) and drops everything else: scripts, styles, iframes, event-handler
attributes, inline styles and javascript:/data: URLs (except data:image for
<img>). Text is always escaped.
"""
from html import escape
from html.parser import HTMLParser
from urllib.parse import urlparse

ALLOWED_TAGS = {
    "p", "br", "hr", "strong", "b", "em", "i", "u", "s", "sub", "sup", "small", "mark",
    "code", "pre", "kbd", "samp", "blockquote", "span", "div",
    "ul", "ol", "li", "dl", "dt", "dd",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption",
    "a", "img",
}
VOID_TAGS = {"br", "hr", "img"}
DROP_CONTENT_TAGS = {"script", "style", "iframe", "object", "embed", "template", "noscript", "svg", "math"}
ALLOWED_ATTRS = {
    "a": {"href", "title"},
    "img": {"src", "alt", "width", "height", "title"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan", "scope"},
    "ol": {"start"},
}


def _safe_url(value, allow_data_image=False):
    value = (value or "").strip()
    scheme = urlparse(value).scheme.lower()
    if scheme in ("http", "https", "mailto", ""):
        # Relative URLs are fine; reject protocol-relative tricks like "//evil" only if they smuggle a scheme.
        return not value.lower().startswith(("javascript", "vbscript"))
    if allow_data_image and scheme == "data":
        return value.lower().startswith(("data:image/png", "data:image/jpeg", "data:image/gif", "data:image/webp"))
    return False


class _Sanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.open = []
        self.drop_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in DROP_CONTENT_TAGS:
            if tag not in VOID_TAGS:
                self.drop_depth += 1
            return
        if self.drop_depth or tag not in ALLOWED_TAGS:
            return
        kept = []
        for name, value in attrs:
            if name not in ALLOWED_ATTRS.get(tag, ()):
                continue
            if name == "href" and not _safe_url(value):
                continue
            if name == "src" and not _safe_url(value, allow_data_image=True):
                continue
            kept.append(f' {name}="{escape(value or "", quote=True)}"')
        if tag == "a":
            kept.append(' rel="noopener noreferrer"')
        self.out.append(f"<{tag}{''.join(kept)}>")
        if tag not in VOID_TAGS:
            self.open.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS and self.open and self.open[-1] == tag:
            self.open.pop()
            self.out.append(f"</{tag}>")

    def handle_endtag(self, tag):
        if tag in DROP_CONTENT_TAGS:
            self.drop_depth = max(0, self.drop_depth - 1)
            return
        if self.drop_depth or tag not in ALLOWED_TAGS or tag in VOID_TAGS or tag not in self.open:
            return
        while self.open:
            current = self.open.pop()
            self.out.append(f"</{current}>")
            if current == tag:
                break

    def handle_data(self, data):
        if not self.drop_depth:
            self.out.append(escape(data, quote=False))

    def result(self):
        self.close()
        while self.open:
            self.out.append(f"</{self.open.pop()}>")
        return "".join(self.out)


def sanitize_html(value):
    if not value:
        return ""
    parser = _Sanitizer()
    parser.feed(str(value))
    return parser.result()
