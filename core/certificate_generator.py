"""
Certificate generation from the SVG template.
Converts SVG → PDF directly via cairosvg (no intermediate PNG step).
"""
import base64
import logging
from io import BytesIO
from xml.sax.saxutils import escape

import cairosvg
import qrcode

from django.conf import settings

logger = logging.getLogger(__name__)

# SVG placeholder → safe fallback text used when the real value is unavailable.
_FALLBACKS = {
    "{NAME}": "STUDENT NAME",
    "{USN}": "NNM000000",
    "{course name}": "CCE e-Lab Programming Course",
    "{dd/mm/yyyy}": "01/01/2025",
    "{FACULTY_NAME}": "FACULTY COORDINATOR",
    "{fac_grade}": "Course Faculty",
}
_UPPERCASE = ("{NAME}", "{USN}", "{FACULTY_NAME}")

# Free space between the two signature blocks (SVG user units, A4 landscape 842.25 x 595.5).
_QR_SIZE = 58
_QR_X = (842.25 - _QR_SIZE) / 2
_QR_Y = 458


def _substitute_placeholders(svg_content: str, student, course_name: str, issued_date: str, faculty_name: str) -> str:
    """Replace every known placeholder in the SVG with the real (XML-escaped) value,
    falling back to a sensible default when the value is empty."""
    values = {
        "{NAME}": student.display_name,
        "{USN}": student.usn or student.username,
        "{course name}": course_name,
        "{dd/mm/yyyy}": issued_date,
        "{FACULTY_NAME}": faculty_name,
        "{fac_grade}": "Course Faculty",
    }
    for key, value in values.items():
        value = str(value or "").strip() or _FALLBACKS[key]
        if key in _UPPERCASE:
            value = value.upper()
        svg_content = svg_content.replace(key, escape(value))
    return svg_content


def _qr_block(verify_url: str) -> str:
    """SVG fragment with a QR code for the public verification page."""
    buffer = BytesIO()
    qrcode.make(verify_url, border=1).save(buffer, "PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<image x="{_QR_X:.2f}" y="{_QR_Y}" width="{_QR_SIZE}" height="{_QR_SIZE}" '
        f'xlink:href="data:image/png;base64,{data}"/>'
        f'<text x="421.125" y="{_QR_Y + _QR_SIZE + 9}" text-anchor="middle" '
        'font-family="sans-serif" font-size="6.5" fill="#3b2a5a">Scan to verify</text>'
    )


def generate_certificate_pdf(student, course, issued_date, verify_url, faculty=None):
    """
    Generate a certificate PDF for `student` completing `course` from the SVG
    template, with a verification QR code. `faculty` (the member who verified
    the request) signs as course faculty; otherwise one managing the course.
    Returns PDF bytes.
    """
    svg_path = str(settings.BASE_DIR / "templates" / "certificates" / "hi.svg")

    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
    except FileNotFoundError:
        logger.error(f"Certificate SVG template not found: {svg_path}")
        raise

    course_name = course.name if course else ""
    faculty_name = _resolve_faculty_name(course, faculty)
    svg_content = _substitute_placeholders(svg_content, student, course_name, issued_date, faculty_name)
    if verify_url:
        svg_content = svg_content.replace("</svg>", _qr_block(verify_url) + "</svg>", 1)

    # A4 landscape: 842.25 x 595.5 points (same as the SVG viewBox).
    try:
        return cairosvg.svg2pdf(
            bytestring=svg_content.encode("utf-8"),
            output_width=842.25,
            output_height=595.5,
        )
    except Exception as exc:
        logger.error(f"cairosvg PDF generation failed: {exc}", exc_info=True)
        raise


def _resolve_faculty_name(course, faculty=None) -> str:
    """Name of the verifying faculty, else a faculty member managing the course, else a fallback."""
    coordinator = faculty
    if coordinator is None and course:
        managers = course.managing_faculty.order_by("id")
        coordinator = managers.filter(role="faculty").first() or managers.first()
    if coordinator:
        return coordinator.display_name or coordinator.username
    return _FALLBACKS["{FACULTY_NAME}"]
