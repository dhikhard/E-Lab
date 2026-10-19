# PRAXIS (CCE e-Lab) - Coding and Placement Skill Development Platform

**PRAXIS** is a self-hosted web platform engineered for the **Department of Computer and Communication Engineering**, NMAM Institute of Technology, Nitte. It delivers continuous coding practice, automated evaluation, progressive AI tutoring, proctored lab assessments, and automated skill certification. Built for high-concurrency campus lab environments, PRAXIS reliably supports up to 400 concurrent students.

---

## 🚀 Key Features

- ✅ **Interactive Code Editor** — Monaco / VS Code editor with syntax highlighting for C, C++, Java, and Python
- ✅ **Automated Evaluation** — Instant feedback via isolated Docker sandbox containers with sub-second execution
- ✅ **Multi-Language Support** — C (GCC C11), C++ (G++ C++17), Java (OpenJDK 17), Python (CPython 3)
- ✅ **Adaptive Question Bank** — Tiered questions (Easy/Medium/Hard) with mandatory problem guarantees; DBMS modules use a single question list
- ✅ **Progress Tracking** — Real-time dashboards showing module completion, scores, and class rankings
- ✅ **Skill Certification** — Student → faculty → HoD approval flow; approved students get a PDF certificate with a QR code to a public verification page
- ✅ **Faculty & HOD Portals** — Create modules/questions, edit questions and their test cases, review submissions, manage certificate approvals
- ✅ **Quizzes & Assignments** — Timed quizzes and take-home open-ended assignments
- ✅ **LeetCode Integration** — Bulk import 1,500+ public problems with test cases from local dataset
- ✅ **RAG Question Generator** — Faculty generate sandbox-validated questions from the DSA notes or their own uploaded material with a local CPU-only LLM; questions are saved for review before going live
- ✅ **Semester Auto-Advance** — Students automatically advance semesters on Jan 1 and Jul 1 via Celery beat
- ✅ **Course Access Control** — Courses unlock based on student semester (C→sem 1, Python/Java/Placement→sem 3, C++/Advanced Placement→sem 5)
- ✅ **Proctoring** — Tab-switch and fullscreen-exit detection stored server-side, in-editor copy/paste allowed, and a focus mode with a Read Problem panel that never triggers violations
- ✅ **AI Tutor Hints** — Three progressive, language-aware hints per question from a local CPU-only LLM, checked for code/answer leaks (100% offline, no external API keys)
- ✅ **Seamless In-Page Auto-Reveal** — Hints automatically reveal with a glowing Cyber Violet & Cyan animation immediately upon test evaluation without page refreshes
- ✅ **Redis-backed Cache** — High-performance caching and session management for 400+ concurrent users

---

## 🏗️ Architecture

- **Backend:** Django 5.0 + Django REST Framework
- **Database:** PostgreSQL 15 (`elab-db`)
- **Cache & Message Broker:** Redis 7 (`elab-redis`)
- **Task Runner:** Celery Worker (`elab-worker`) with prefork pool
- **Scheduler:** Celery Beat (`elab-beat`) for periodic tasks
- **Execution Sandbox:** Custom isolated Docker container (`elab-sandbox`) supporting C, C++, Java, and Python
- **Reverse Proxy:** Nginx (`elab-nginx`) with gzip compression and keepalive
- **RAG Engine:** Structure-aware chunks + Ollama embeddings stored in PostgreSQL, hybrid semantic/keyword retrieval
- **Local AI (CPU-only):** Ollama (`elab-ollama`, `qwen3.5:4b` + `nomic-embed-text`) on a dedicated `worker-llm` Celery queue, with deterministic fallbacks
- **Frontend:** Responsive HTML5 templates + Monaco Editor + Cyber Violet styling & CSS animations

---

## 📋 Prerequisites

| Software | Minimum Version | Purpose |
|----------|-----------------|---------|
| Docker Engine / Desktop | 24.0+ | Container runtime & execution sandbox |
| Docker Compose | 2.20+ | Multi-container orchestration |
| Git | 2.30+ | Version control |
| curl | — | Health checks |

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/kevincell/E-Lab.git
cd E-Lab
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
```

Edit `.env` and set secure values:
```ini
# Django Configuration
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1,your.domain.com

# Database Configuration
POSTGRES_DB=elab_db
POSTGRES_USER=elab
POSTGRES_PASSWORD=change-this-password

# Redis & Celery
REDIS_URL=redis://elab-redis:6379/0
CELERY_BROKER_URL=redis://elab-redis:1
CELERY_RESULT_BACKEND=redis://elab-redis:2

# Sandbox Settings
DOCKER_SANDBOX_IMAGE=elab-sandbox
DOCKER_SANDBOX_DIR=/var/elab-sandbox
HOST_SANDBOX_DIR=/home/nmamit/e_lab/E-Lab/sandbox_data

# Local LLM (CPU-only, no external API keys) — see "AI Tutor Hints" for all options
OLLAMA_URL=http://elab-ollama:11434/api/chat
HINT_MODEL=qwen3.5:4b
OLLAMA_THINK=false
OLLAMA_NUM_THREAD=6
OLLAMA_NUM_CTX=6144

# HOD account created by setup.sh / create_hod (no default password)
HOD_PASSWORD=choose-a-strong-password

# Email (optional - for certificate notifications)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your@email.com
EMAIL_HOST_PASSWORD=your-app-password
EMAIL_USE_TLS=True
DEFAULT_FROM_EMAIL=noreply@yourdomain.com
```

### 3. Build Sandbox & Start Services
```bash
# 1. Build the multi-language execution sandbox image
docker build -t elab-sandbox -f sandbox/Dockerfile sandbox/

# 2. Start all services (Django, Postgres, Redis, Celery, Nginx, Ollama)
docker compose up -d --build

# 3. Wait for services to be healthy
docker compose ps
```

### 4. Initialize Database, Static Files & Questions
```bash
# Migrations and collectstatic also run automatically when the app container starts
docker compose exec app python manage.py migrate
docker compose exec app python manage.py collectstatic --noinput

# Create the HoD account (uses HOD_PASSWORD from .env if --password is omitted)
docker compose exec app python manage.py create_hod --password '<choose-a-strong-password>'

# Seed course catalog (C, Python, Java, C++, Placement Training)
docker compose exec app python manage.py seed_courses

# Import default question bank (creates courses and modules)
docker compose exec app python manage.py import_questions

# Bulk import 1,500 LeetCode questions with test cases
docker compose exec app python scripts/import_leetcode_problems.py

# Enrich all questions to ~8 test cases each
docker compose exec app python manage.py enrich_test_cases

# Seed demo users (faculty and first/second/third-year students)
docker compose exec app python manage.py seed_demo
```

### 5. Pull the CPU models (one-time) and build the question-generation index
The server has no GPU; everything below runs on the CPU.
```bash
# Hint + question-generation model (~3.4 GB) and embedding model (~0.3 GB)
docker compose exec ollama ollama pull qwen3.5:4b
docker compose exec ollama ollama pull nomic-embed-text

# Chunk + embed the bundled DSA notes (~2 minutes on an 8-core i7). Optional: the notes are
# also loaded automatically the first time the AI Question Agent is opened, and embedded in the
# background; until then generation ranks them by keywords.
docker compose exec worker python manage.py rag_index

# Optional: pre-generate tutor hints before a lab (~25 s per question per language)
docker compose exec worker python manage.py pregenerate_hints --module-id <id>
```

### 6. Access the Application
The web app is available at `http://localhost` (or your server IP):

| URL | Description |
|-----|-------------|
| `http://localhost` | Main Portal / Student Dashboard |
| `http://localhost/login/` | Authentication Page |
| `http://localhost/admin/` | Django Admin Panel |
| `http://localhost/health/` | Health Check Endpoint |

---

## 🔑 Demo & Test Accounts

Created via `docker compose exec app python manage.py seed_demo` (for demos only — never on a production server).
The HoD account is not part of the demo data; create it with `create_hod` (step 4).

| Role | Username | Password | Notes |
|------|----------|----------|-------|
| **Faculty (CS)** | `faculty_cs` | `faculty123` | Module & question management |
| **Faculty (IT)** | `faculty_it` | `faculty123` | Module & question management |
| **Student (1st Yr)** | `student_fy_01` | `student123` | Semester 1 student |
| **Student (1st Yr)** | `student_fy_02` | `student123` | Semester 1 student |
| **Student (1st Yr)** | `student_fy_03` | `student123` | Semester 2 student |
| **Student (2nd Yr)** | `student_sy_01` | `student123` | Semester 3 student |
| **Student (2nd Yr)** | `student_sy_02` | `student123` | Semester 3 student |
| **Student (2nd Yr)** | `student_sy_03` | `student123` | Semester 4 student |
| **Student (3rd Yr)** | `student_ty_01` | `student123` | Semester 5 student |
| **Student (3rd Yr)** | `student_ty_02` | `student123` | Semester 5 student |
| **Student (3rd Yr)** | `student_ty_03` | `student123` | Semester 6 student |

---

## 🛠️ Management Commands Reference

All commands should be executed against the `app` container:
```bash
docker compose exec app python manage.py <command> [options]
```

### Seed Demo Data
Populates faculty, students, sample submissions, and student progress records:
```bash
docker compose exec app python manage.py seed_demo
```

### Course & Curriculum Seeding
Populate the database with predefined courses, modules, and questions:
```bash
# Seed Course Catalog (C, Python, Java, C++, Placement Training, Advanced Placement Training)
docker compose exec app python manage.py seed_courses

# Import first-year questions (C Programming - default)
docker compose exec app python manage.py import_questions

# Seed Java Programming course (Lab manual & RAG generated pool)
docker compose exec -T app python manage.py shell < scripts/reseed_java.py

# Seed C++ Programming course (15 modules + LeetCode pool bypass)
docker compose exec -T app python manage.py shell < scripts/reseed_cpp.py
docker compose exec app python scripts/seed_leetcode_cpp.py

# Seed Python Programming course (15 modules + LeetCode pool bypass)
docker compose exec app python scripts/reseed_python.py
docker compose exec app python scripts/seed_leetcode_python.py

# Import second-year placement questions
docker compose exec app python manage.py seed_placement_training

# Import third-year advanced placement questions
docker compose exec app python manage.py seed_advanced_placement_training

# Create the DBMS (SQL) course: 3 modules × 5 questions (replaces its modules if it exists)
docker compose exec app python scripts/create_dbms_course.py
```

### Certificates
Certificates are issued only through the approval flow (see **Certificate Flow** below). This command issues
certificates for HoD-approved requests that don't have one yet and renders missing PDFs:
```bash
# Backfill missing certificates / PDFs
docker compose exec app python manage.py generate_certificates

# Preview without changing anything
docker compose exec app python manage.py generate_certificates --dry-run

# Re-render every certificate PDF (e.g. after editing templates/certificates/hi.svg)
docker compose exec app python manage.py generate_certificates --regenerate
```

### Import Questions from LeetCode
```bash
# Full bulk import (idempotent, skips existing)
docker compose exec app python scripts/import_leetcode_problems.py

# Import a single question by slug
docker compose exec app python manage.py import_leetcode --question two-sum --module "LeetCode Problems" --difficulty easy --csv-level 1

# Import by numeric ID
docker compose exec app python manage.py import_leetcode --question 1 --module "LeetCode Problems" --difficulty easy
```

### Enrich Questions with More Test Cases
```bash
# Enrich all DSA module questions to ~8 test cases each
docker compose exec app python manage.py enrich_test_cases

# Test on first 10 questions
docker compose exec app python manage.py enrich_test_cases --limit 10

# Dry-run to preview changes
docker compose exec app python manage.py enrich_test_cases --dry-run

# Custom target (default: 8)
docker compose exec app python manage.py enrich_test_cases --target 10
```

### Clean & Format Test Cases
```bash
# Clean up LeetCode JSON test cases (removes commas and brackets to match standard competitive programming formats)
docker compose exec app python scripts/clean_testcases.py
```

### Semester Management
```bash
# Auto-advance student semesters (only acts on Jan 1 and Jul 1)
docker compose exec app python manage.py auto_advance_semesters

# Dry-run to see what would happen
docker compose exec app python manage.py auto_advance_semesters --dry-run
```

### Create Users
```bash
# Create the HoD user (default username: hod)
docker compose exec app python manage.py create_hod --password '<choose-a-strong-password>'

# Create Django Superuser (for admin panel)
docker compose exec app python manage.py createsuperuser

# Create a student user
docker compose exec app python manage.py shell -c "
from core.models import User
u = User.objects.create_user(username='NNM25CC001', usn='NNM25CC001', password='<password>', role='student', department='CCE', semester=3)
print(f'Created: {u}')
"
```

---

## 🤖 Question Generator (RAG + local LLM, CPU-only)

Faculty open the **AI Question Agent** drawer (bottom-right on any faculty page), pick a topic, language,
difficulty (or *Mixed*) and how many questions, and optionally **upload their own notes** (`.md`, `.txt` or
text-based `.pdf`, up to 5 MB). Generation runs as a background job on the `worker-llm` queue; the drawer
polls and shows progress.

### Pipeline
1. **Chunking** — documents are split by structure (headings, paragraphs, lists); fenced code blocks are never
   split; ~1,200-character chunks with one-paragraph overlap.
2. **Retrieval** — `nomic-embed-text` embeddings (stored with model + version; `rag_index` re-embeds stale chunks)
   blended with a keyword score, then de-duplicated. On 20 labelled queries, precision@3 went from 0.37 (old keyword
   matcher) to 0.68. The bundled `data/DSA_Topics` notes are loaded automatically on first use and embedded in the
   background; until then (or while the embedding model is down) chunks are ranked by keywords only.
3. **Generation** — one schema-constrained LLM call per question, with the retrieved chunks, topic, language,
   explicit easy/medium/hard criteria and an output schema (statement, formats, constraints, starter code,
   reference solution, 5–6 tests, citations). The fixed part of the prompt (rules + sources) comes first so
   Ollama reuses its prompt cache for later questions in a job and for retries.
4. **Validation** — a candidate is discarded unless: the JSON matches the schema; every citation is a retrieved
   chunk and the statement shares vocabulary with it; the code is in the requested language and the starter code
   doesn't contain the solution; **the reference solution, run in the Docker sandbox, prints every expected
   output** and the starter code compiles; and it isn't a near-duplicate (embedding similarity ≥ 0.92, or a matching
   title if embeddings are unavailable) of an existing question in the same course. One retry with the failure fed back is allowed. A heuristic flags suspicious difficulty labels.
5. **Review** — accepted questions are saved **inactive** in the module they were generated for. Faculty review,
   edit (including test cases) and activate them from the question bank.

```bash
# CLI equivalents (also save inactive)
docker compose exec worker python manage.py generate_question --topic "stack" --difficulty easy --language c --module-id 3
docker compose exec worker python manage.py bulk_generate_questions --topics "stack,binary search" --difficulty medium --module-id 3
# Evaluate on chosen source documents (writes everything for manual grading)
docker compose exec worker python manage.py eval_question_gen --doc 04_Stack/01_valid_parentheses.md:python --count 3
```
Results of the evaluation are in `core/evals/README.md`.

---

## 🎓 Certificate Flow

1. **Student** — once a course reaches the threshold (`CERTIFICATE_THRESHOLD`, default 80%) with every mandatory
   question solved, the course dashboard shows **Get Certificate**, which creates a request for that course.
   Completion counts up to 12 questions per module (5 for DBMS modules, 7 for placement training, which issues no
   certificates).
2. **Faculty** — the faculty managing the course (all faculty if nobody manages it) are notified and forward the
   request to the HoD from **Cert Requests** or the student's profile page, with optional remarks.
3. **HoD** — reviews the course progress and submissions, then approves (the certificate is issued immediately) or
   declines with remarks. A declined student can re-apply after solving at least one more question.
4. **Certificate** — a PDF rendered from `templates/certificates/hi.svg` with the student's name and USN, the course,
   the issue date, the verifying faculty and a QR code to the public `/verify/<hash>/` page. Students and faculty
   download it from the certificate page.

---

## 💡 AI Tutor Hints (3 progressive hints, no answers)

- **One local LLM call per (question, language)** returns exactly three hints as JSON: level 1 helps the student
  understand the problem, level 2 asks a question pointing at the key property, level 3 names the most common
  pitfall *in the student's language* (e.g. `int` overflow in Java, `input()` returning a string in Python).
- **Leak guard**: hints with code, backticks, function signatures, assignments, step lists, “the answer is”,
  a hidden test's expected output, a named technique (stack, hash map, DP, DFS, sliding window, …) or a construct
  from another language are rejected; the model gets one retry, then a deterministic language-aware fallback is used.
  Describing the method in plain words can't be caught by rules — see `core/evals/README.md` for measured results.
- **Cached** per (question, language, difficulty, code-hash) in Redis and the `GeneratedHintSet` table (editable in
  Django admin so faculty can review). The web request never waits for the model: on a cache miss the student gets
  the fallback immediately and the hint upgrades when the LLM set is ready (the UI announces it).
- **Limits**: 3 hints per question; 10 unlocks per student per minute; 90 s model timeout; 5-minute back-off after a
  failure. Prompts and responses are logged to the `core.hints.llm` logger without student code.
- **Hints never reduce points.**

| Setting | Default | Meaning |
|---|---|---|
| `HINT_MODEL` | `qwen3.5:4b` | Ollama model for hints |
| `OLLAMA_THINK` | `false` | Disable hidden reasoning tokens (much faster on CPU) |
| `OLLAMA_NUM_THREAD` | `6` (`.env.example`) | Threads per generation |
| `HINT_LLM_TIMEOUT` | `90` | Seconds before falling back |
| `HINTS_USE_STUDENT_CODE` | `false` | Send the student's own code (more specific, much less cacheable) |
| `QUESTION_GEN_MODEL` | `HINT_MODEL` | Model for question generation |
| `OLLAMA_NUM_CTX` | `6144` | Context window for every model request; one value for hints and generation so the model is never reloaded |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for retrieval and de-duplication |

---

## 🏆 Points & Ranking

- **Easy 10 · Medium 25 · Hard 50** points on the *first* accepted solution of a question.
- Each failed submission before that (wrong answer, TLE, runtime or compile error) removes 10%, down to 50% of base.
  Infrastructure errors don't count. Re-solving earns nothing. **Hints never reduce points.**
- Points are computed server-side when a submission is judged (or manually accepted) and stored in a per-question
  ledger (`PointsLedger`), so every award is auditable. `python manage.py recalculate_points` rebuilds it from history.
- **Ranking**: total points, ties broken by hard problems solved, then medium, then who reached the score first.
  Tiers: Bronze (0), Silver (100), Gold (300), Platinum (700). Students see a per-problem breakdown on the
  leaderboard and on each accepted submission.

---

## 🛡️ Proctoring & Focus Mode

- **What counts as a violation**: the tab becoming hidden (`visibilitychange`, plus a debounced `blur` check),
  leaving browser fullscreen, and DevTools shortcuts (F12, Ctrl+Shift+I/J/C, Ctrl+U). Copy/cut/drag and right-click
  outside the editor are blocked with a notice but not counted.
- **Focus mode** (click the editor or *Focus mode*): a pure CSS layout state — it never calls the browser
  Fullscreen API, so entering/leaving it or toggling **Read Problem** (`Ctrl+\`) can't trigger a violation.
  *Exit focus mode* or `Esc` returns to the page; focus is kept inside the overlay and returned to the editor.
  On narrow screens the problem, hints and editor stack vertically.
- **Browser fullscreen belongs to proctoring**: it is requested when the assessment starts (and re-requested on the
  next click if lost); leaving it is always a violation. In Chromium, Keyboard Lock means a short `Esc` press closes
  focus mode and a long press is needed to leave fullscreen; in other browsers `Esc` leaves fullscreen (logged).
- **Server-side record**: violations and focus-mode/read-problem transitions are stored in `ProctoringEvent`
  (not the session), attached to the next submission, rate-limited, and shown to faculty as a timeline on the
  submission page (UI transitions are marked “not a violation”).
- Graduated notices: three toasts, then a modal.

---

## 📦 Docker Sandbox Execution

The custom Docker sandbox executes code submissions in fully isolated containers:

| Language | `language_id` | Toolchain | Strictness / Options |
|----------|---------------|-----------|----------------------|
| **C** | 50 | GCC (`gcc -std=c11`) | `-Wall -Wextra -Werror=return-type -Werror=main` |
| **C++** | 54 | GCC (`g++ -std=c++17`) | `-Wall -Wextra -Werror=return-type -Werror=main` |
| **Java** | 62 | OpenJDK 17 (`javac`/`java`) | Auto-detects public class name |
| **Python** | 71 | CPython 3 (`python3`) | Direct execution, syntax-checked |
| **SQL** | 82 | SQLite | DBMS modules; the test input sets up the tables |

### Sandbox Security Flags
- **No network** (`--network none`), **all capabilities dropped**, `no-new-privileges`, **runs as `nobody`**
- **Read-only root filesystem**; the per-run host directory is mounted **read-only**; `/tmp` is a 50 MB tmpfs
- **Limits**: CPU 0.5, 64 PIDs, 256 open files, 16 MB per file, stdout capped at 1 MB and stderr at 64 KB
  (a flood ends the run as a runtime error instead of exhausting worker memory)
- **Memory**: the question's limit (default 125 MB); for C/C++ the container gets 384 MB so the compiler fits
  (`<bits/stdc++.h>` needs ~250 MB) and the question's limit is applied to the program with `ulimit -v`
- Code runs on the Celery workers; **the web container has no Docker socket** (`SANDBOX_VIA_WORKER=true`)
- Source and stdin are base64-encoded into the run script; fresh container per run

---

## 🔧 Useful Docker & Maintenance Commands

```bash
# View live logs
docker compose logs -f app
docker compose logs -f worker
docker compose logs -f worker-llm
docker compose logs -f elab-db

# Restart services after configuration change
docker compose restart app worker worker-llm nginx

# Stop all services
docker compose down

# Stop and remove volumes (WARNING: deletes all data)
docker compose down -v

# Run test suite
docker compose exec app python manage.py test

# Backup database
docker compose exec elab-db pg_dump -U elab elab_db > backup_$(date +%Y%m%d).sql

# Restore database
docker compose exec -T elab-db psql -U elab elab_db < backup_20250101.sql

# Regenerate static files
docker compose exec app python manage.py collectstatic --noinput

# Check container health
docker compose ps

# Access PostgreSQL shell
docker compose exec elab-db psql -U elab -d elab_db

# Access Redis CLI
docker compose exec elab-redis redis-cli
```

---

## 📊 Capacity on the 16 GB, CPU-only server

| Service | Memory limit | Notes |
|---|---|---|
| `app` (gunicorn 16×4 gthread) | 2.5 GB | no Docker socket |
| `worker` (judge/run, concurrency 12) | 2 GB | each sandbox container: 125 MB (384 MB for C/C++), 0.5 CPU |
| `worker-llm` (concurrency 1) | 1 GB | hints, question generation, embeddings |
| `ollama` (1 request at a time) | 5 GB | qwen3.5:4b (~3.5 GB) + nomic-embed-text |
| `elab-db` | 2 GB | `max_connections=500`, `shared_buffers=256MB` |
| `elab-redis` | 768 MB | `maxmemory 512mb` |
| `nginx` | 256 MB | |

Sandbox containers run outside these limits: typical runs use a few MB, but 12 simultaneous large C++ compiles could
use ~3 GB, which is why worker concurrency is 12 rather than 32. Measured on an i7‑11800H with 6 threads:
hint sets ~23 s, embeddings for the whole corpus ~100 s, question generation several minutes per question.

---

## 🔍 Troubleshooting

### 1. `DisallowedHost` or `Bad Request (400)`
- Ensure `ALLOWED_HOSTS` in `.env` includes `localhost,127.0.0.1` and your server's domain/IP.
- Restart app: `docker compose restart app`.

### 2. Database Connection Error
- Ensure container `elab-db` is running: `docker compose ps`.
- Check database logs: `docker compose logs elab-db`.
- Verify database credentials in `.env` match `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD`.

### 3. Static Files Not Loading / Unstyled UI
- Run static collection:
  ```bash
  docker compose exec app python manage.py collectstatic --noinput
  docker compose restart nginx
  ```

### 4. Sandbox Execution Errors
- Ensure the sandbox image is built:
  ```bash
  docker build -t elab-sandbox -f sandbox/Dockerfile sandbox/
  ```
- Code runs on the `worker` container (the web container has no Docker socket); check `docker compose logs worker`
- Check sandbox directory permissions: `chmod 777 sandbox_data`

### 5. Celery Worker Not Processing Tasks
- Check worker logs: `docker compose logs -f worker`
- Verify Redis is reachable: `docker compose exec elab-redis redis-cli ping`
- Restart worker: `docker compose restart worker`

### 6. Question Generator Issues
- "No source material is indexed yet": the `data/DSA_Topics` notes are missing from the image; or run `docker compose exec worker python manage.py rag_index`
- Jobs stuck in "queued": check the LLM worker — `docker compose logs -f worker-llm`
- Every candidate rejected: the drawer lists the stage (schema, citation, grounding, language, execution, duplicate)

### 7. Certificate PDF Generation Fails
- PDFs are rendered with CairoSVG; check `docker compose logs app` for "PDF generation failed"
- The certificate page retries a missing PDF on every visit; `python manage.py generate_certificates` retries all of them
- Make sure the `media` volume is writable by the app container

### 8. Local AI / Tutor Hints Troubleshooting
- **Ollama container status**: Verify Ollama is running: `docker compose ps ollama`.
- **Model verification**: Ensure `qwen3.5:4b` and `nomic-embed-text` are downloaded:
  ```bash
  docker compose exec ollama ollama list
  ```
- **Fallback verification**: If Ollama is downloading, cold, or stopped, PRAXIS automatically falls back to the deterministic diagnostic engine. No student submissions fail or hang.

---

## 📄 License

Academic use — Department of Computer & Communication Engineering (CCE), NMAM Institute of Technology, Nitte.
