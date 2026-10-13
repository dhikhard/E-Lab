from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Deprecated alias for `rag_index` (the old ChromaDB index is no longer used by the app)."

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING("rag_ingest is deprecated; running rag_index instead."))
        call_command("rag_index")
