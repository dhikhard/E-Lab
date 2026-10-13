from django import template
register = template.Library()

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)


@register.filter(is_safe=True)
def safe_html(value):
    """Allowlist-sanitised HTML for authored descriptions (replaces |safe)."""
    from django.utils.safestring import mark_safe

    from core.html_sanitizer import sanitize_html

    return mark_safe(sanitize_html(value))
