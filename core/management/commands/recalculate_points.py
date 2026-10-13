from django.core.management.base import BaseCommand

from core.points import FORMULA_VERSION, recalculate_all


class Command(BaseCommand):
    help = "Rebuild the points ledger from submission history using the current formula."

    def handle(self, *args, **options):
        created = recalculate_all()
        self.stdout.write(self.style.SUCCESS(f"Ledger rebuilt (formula v{FORMULA_VERSION}): {created} awards."))
