from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.templatetags.static import static as static_url
from django.urls import include, path
from django.views.generic import RedirectView

urlpatterns = [
    path("admin/", admin.site.urls),
    # Browsers request /favicon.ico on pages without an icon link (e.g. Django admin).
    path("favicon.ico", RedirectView.as_view(url=static_url("img/Logologo.png"), permanent=True)),
    path("", include("core.urls")),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
