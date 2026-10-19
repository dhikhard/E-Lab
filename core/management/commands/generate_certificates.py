#!/usr/bin/env python
"""
Issue certificates for HoD-approved requests that don't have one yet, and
render missing PDFs. Certificates are only ever issued through the
student → faculty → HoD approval flow; this command never bypasses it.

    python manage.py generate_certificates             # backfill missing certificates/PDFs
    python manage.py generate_certificates --regenerate # re-render every PDF (e.g. after a template change)
"""

from django.core.management.base import BaseCommand

from core.models import Certificate, CertificateRequest
from core.services import issue_certificate


class Command(BaseCommand):
    help = "Issue certificates for approved requests and (re-)render certificate PDFs."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="List what would be done without changing anything.")
        parser.add_argument("--regenerate", action="store_true", help="Re-render the PDF of every existing certificate.")

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        rendered = failed = 0

        approved = (
            CertificateRequest.objects.filter(status=CertificateRequest.Status.APPROVED, course__isnull=False)
            .select_related("student", "course", "requested_by_faculty")
            .order_by("updated_at")
        )
        for req in approved:
            cert = Certificate.objects.filter(student=req.student, course=req.course).first()
            if cert and cert.pdf and cert.qr_code and not options["regenerate"]:
                continue
            label = f"{req.student.username} — {req.course.name}"
            if dry_run:
                self.stdout.write(f"  would {'render' if cert else 'issue'}: {label}")
                continue
            if cert and options["regenerate"] and cert.pdf:
                cert.pdf.delete(save=True)  # issue_certificate re-renders a missing PDF
            cert = issue_certificate(req.student, req.course, req.completion_percentage, req.requested_by_faculty)
            if cert.pdf:
                rendered += 1
                self.stdout.write(self.style.SUCCESS(f"  ✔ {label}"))
            else:
                failed += 1
                self.stdout.write(self.style.ERROR(f"  ✗ PDF failed: {label} (see logs)"))

        self.stdout.write(f"\nDone: {rendered + failed} processed, {rendered} with PDF, {failed} failed{' (dry run)' if dry_run else ''}.")
