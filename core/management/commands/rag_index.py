from django.core.management.base import BaseCommand

from core.rag import retrieval


class Command(BaseCommand):
    help = "Load the bundled DSA corpus into the RAG index and (re-)embed stale chunks."

    def add_arguments(self, parser):
        parser.add_argument("--skip-corpus", action="store_true", help="Only re-embed existing chunks")

    def handle(self, *args, **options):
        if not options["skip_corpus"]:
            seen, changed = retrieval.index_corpus()
            self.stdout.write(f"Corpus: {seen} documents, {changed} new or changed.")
        stale = retrieval.stale_chunks().count()
        self.stdout.write(f"Embedding {stale} chunk(s) with {retrieval.EMBED_MODEL} (v{retrieval.EMBED_VERSION})...")

        def progress(done, total):
            if done == total or done % 128 == 0:
                self.stdout.write(f"  {done}/{total}")

        retrieval.embed_pending(progress=progress)
        self.stdout.write(self.style.SUCCESS("RAG index is up to date."))
