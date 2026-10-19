from django.conf import settings
from django.core.cache import cache


def site_settings(request):
    ctx = {
        "SITE_NAME": settings.SITE_NAME,
        "CERTIFICATE_THRESHOLD": settings.CERTIFICATE_THRESHOLD,
    }
    if hasattr(request, "user") and request.user.is_authenticated:
        cache_key = f"unread_notif_{request.user.id}"
        count = cache.get(cache_key)
        if count is None:
            from .models import Notification
            count = Notification.objects.filter(
                recipient=request.user, is_read=False
            ).count()
            cache.set(cache_key, count, 15)
        ctx["unread_notification_count"] = count
    return ctx

