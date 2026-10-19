# Ad-hoc debug scripts

One-off scripts used while debugging specific pages and submissions. They were
`test_*.py` files in the repository root, where `python manage.py test` imported
them during test discovery and ran their queries (and writes) against the
configured database before the test database existed. They are not tests.

Run one from the repository root against a database you're happy to touch:

```bash
PYTHONPATH=. python scripts/adhoc/debug_dash.py
```

Several assume specific rows (e.g. question 2699, user `student_test`).
