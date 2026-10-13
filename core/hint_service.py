"""
Student-facing hint flow (T3).

Hints are three progressive, LLM-generated, leak-checked hints per
(question, language), produced by core.llm_hints in a Celery task and stored
in Redis (L1) and GeneratedHintSet (L2, durable). Unlocking never waits on the
model: if the set isn't ready yet the student gets a deterministic,
language-aware fallback immediately (hint_type "diagnostic"), and that row is
upgraded when the LLM set arrives (the UI announces the upgrade).

Hints never affect points (see core.points).
"""
import json
import logging
import os

from django.core.cache import cache

from . import llm_hints
from .models import GeneratedHintSet, StudentQuestionHint, Submission
from .sandbox import language_for_id

logger = logging.getLogger(__name__)

MAX_HINTS_PER_QUESTION = 3
UNLOCKS_PER_MINUTE = 10  # per student, across all questions (each question is also capped at 3)
# Sending the student's own code to the model makes hints more specific but
# much less cacheable (one CPU generation per distinct code). Off by default.
HINTS_USE_STUDENT_CODE = os.environ.get("HINTS_USE_STUDENT_CODE", "false").lower() == "true"
SUPPORTED_LANGUAGES = ("c", "cpp", "python", "java")

_INFLIGHT_KEY = "hints:inflight:{key}"
_FAILED_KEY = "hints:failed:{key}"


def resolve_language(question, language_id=None):
    """Language for hints: the student's choice if the question allows it, else the question's language."""
    default = language_for_id(question.language_id)
    if language_id and question.allow_multiple_languages:
        try:
            chosen = language_for_id(int(language_id))
        except (TypeError, ValueError):
            chosen = default
        if chosen in SUPPORTED_LANGUAGES:
            return chosen
    return default if default in SUPPORTED_LANGUAGES else llm_hints.DEFAULT_LANGUAGE


def _code_digest(student, question):
    if not HINTS_USE_STUDENT_CODE or student is None:
        return "-", None
    latest = Submission.objects.filter(student=student, question=question).order_by("-id").first()
    code = latest.code if latest else None
    return llm_hints.code_hash(code), code


def get_hint_set(question, language, code_digest="-"):
    """Return the cached/persisted list of 3 LLM hints, or None."""
    key = llm_hints.cache_key(question, language, code_digest)
    hints = cache.get(key)
    if hints:
        return hints
    row = GeneratedHintSet.objects.filter(
        question=question,
        language=language,
        difficulty=question.difficulty,
        code_hash=code_digest,
        prompt_version=llm_hints.PROMPT_VERSION,
    ).first()
    if row:
        cache.set(key, row.hints, llm_hints.CACHE_TTL)
        return row.hints
    return None


def generation_status(question, language, code_digest="-"):
    key = llm_hints.cache_key(question, language, code_digest)
    if get_hint_set(question, language, code_digest):
        return "complete"
    if cache.get(_INFLIGHT_KEY.format(key=key)):
        return "generating"
    if cache.get(_FAILED_KEY.format(key=key)):
        return "unavailable"
    return "pending"


def enqueue_hint_generation(question, language, code_digest="-", student_id=None):
    """Queue one background generation per (question, language, code) — deduplicated."""
    key = llm_hints.cache_key(question, language, code_digest)
    if cache.get(_FAILED_KEY.format(key=key)):
        return False  # model recently failed for this key; fallback stays until backoff expires
    if not cache.add(_INFLIGHT_KEY.format(key=key), 1, llm_hints.HINT_TIMEOUT * llm_hints.HINT_ATTEMPTS + 60):
        return False
    from .tasks import generate_hint_set_task

    generate_hint_set_task.apply_async(args=[question.id, language, code_digest, student_id], expires=600)
    return True


def store_hint_set(question, language, code_digest, hints):
    GeneratedHintSet.objects.update_or_create(
        question=question,
        language=language,
        difficulty=question.difficulty,
        code_hash=code_digest,
        prompt_version=llm_hints.PROMPT_VERSION,
        defaults={"hints": hints, "model_name": llm_hints.HINT_MODEL},
    )
    cache.set(llm_hints.cache_key(question, language, code_digest), hints, llm_hints.CACHE_TTL)
    # Upgrade fallback rows students already unlocked for this question+language.
    # The UI polls and announces "AI Tutor hint updated", so this is not silent.
    for tier, text in enumerate(hints, start=1):
        StudentQuestionHint.objects.filter(
            question=question, language=language, hint_number=tier, hint_type="diagnostic"
        ).update(hint_text=text, hint_type="local_llm")


def mark_generation_finished(question, language, code_digest, failed):
    key = llm_hints.cache_key(question, language, code_digest)
    cache.delete(_INFLIGHT_KEY.format(key=key))
    if failed:
        cache.set(_FAILED_KEY.format(key=key), 1, llm_hints.FAILURE_BACKOFF)


def get_student_hints_for_question(student, question):
    """
    Returns existing unlocked hints for a student and question, along with metadata.
    """
    if not student.is_authenticated:
        return {
            "hints": [],
            "unlocked_count": 0,
            "max_hints": MAX_HINTS_PER_QUESTION,
            "can_unlock_more": False,
        }

    hints = list(
        StudentQuestionHint.objects.filter(student=student, question=question).order_by("hint_number")
    )
    count = len(hints)
    return {
        "hints": hints,
        "unlocked_count": count,
        "max_hints": MAX_HINTS_PER_QUESTION,
        "can_unlock_more": count < MAX_HINTS_PER_QUESTION,
        "next_hint_number": count + 1 if count < MAX_HINTS_PER_QUESTION else None,
    }


def can_unlock_now(student):
    """Per-student budget so hint unlocks (and the LLM jobs they queue) can't be spammed."""
    key = f"hint-unlock:{student.id}"
    cache.add(key, 0, 60)
    try:
        return cache.incr(key) <= UNLOCKS_PER_MINUTE
    except ValueError:
        return True


def unlock_hint_for_question(student, question, language=None):
    """
    Unlock the next progressive hint (1..3) for `language`. Returns immediately:
    the LLM hint if it's ready, otherwise a language-aware fallback, while the
    LLM set is generated in the background. Capped at MAX_HINTS_PER_QUESTION.
    """
    if not student or not student.is_authenticated:
        return None

    existing_count = StudentQuestionHint.objects.filter(student=student, question=question).count()
    if existing_count >= MAX_HINTS_PER_QUESTION:
        logger.info(f"Student {student.id} already has {existing_count} hints on Question {question.id}. Max reached.")
        return None
    next_hint_num = existing_count + 1

    existing = StudentQuestionHint.objects.filter(
        student=student, question=question, hint_number=next_hint_num
    ).first()
    if existing:
        return existing

    language = language or resolve_language(question)
    code_digest, _code = _code_digest(student, question)
    hints = get_hint_set(question, language, code_digest)
    if hints:
        llm_hints.bump("cache_hit")
        hint_text, hint_type = hints[next_hint_num - 1], "local_llm"
    else:
        llm_hints.bump("cache_miss")
        llm_hints.bump("fallback_used")
        enqueue_hint_generation(question, language, code_digest, student.id if HINTS_USE_STUDENT_CODE else None)
        hint_text = llm_hints.fallback_hints(question, language)[next_hint_num - 1]
        hint_type = "diagnostic"

    latest_sub = Submission.objects.filter(student=student, question=question).order_by("-id").first()
    try:
        hint_obj, _ = StudentQuestionHint.objects.get_or_create(
            student=student,
            question=question,
            hint_number=next_hint_num,
            defaults={
                "submission": latest_sub,
                "hint_text": hint_text,
                "hint_type": hint_type,
                "language": language,
            },
        )
        return hint_obj
    except Exception as exc:
        logger.error(f"Failed to save StudentQuestionHint: {exc}", exc_info=True)
        return None


def pregenerate_hints_for_question(question, force=False, languages=None):
    """
    Queue LLM hint generation for a question in each language students may use,
    so hints are ready before a lab. Returns {language: status}.
    """
    if languages is None:
        languages = SUPPORTED_LANGUAGES if question.allow_multiple_languages else (resolve_language(question),)
    results = {}
    for language in languages:
        key = llm_hints.cache_key(question, language)
        if force:
            cache.delete(key)
            cache.delete(_FAILED_KEY.format(key=key))
            GeneratedHintSet.objects.filter(
                question=question, language=language, prompt_version=llm_hints.PROMPT_VERSION
            ).delete()
        elif get_hint_set(question, language):
            results[language] = "cached"
            continue
        results[language] = "queued" if enqueue_hint_generation(question, language) else "in-progress"
    return results


def generate_hint_for_submission(submission):
    """
    Backwards-compatible wrapper: delegates to unlock_hint_for_question.
    """
    if not submission or submission.status == Submission.Status.ACCEPTED:
        return None
    return unlock_hint_for_question(
        submission.student, submission.question, resolve_language(submission.question, submission.language_id)
    )


def _extract_failed_test_info(judge_output):
    """Extracts first failing test case stdin and expected output."""
    if not judge_output:
        return "Hidden test case evaluation failed"
    try:
        tests = json.loads(judge_output) if isinstance(judge_output, str) else judge_output
        if isinstance(tests, list):
            for t in tests:
                if not t.get("passed"):
                    stdin = str(t.get("stdin", ""))[:60]
                    expected = str(t.get("expected", ""))[:60]
                    actual = str(t.get("actual", ""))[:60]
                    return f"Input '{stdin}' -> Expected '{expected}', Got '{actual}'"
    except Exception:
        pass
    return "Test case output mismatch"


def _detect_question_topic(question):
    """Detects problem domain / data structure topic from question context."""
    text = f"{question.title} {question.description} {question.module.name if question.module else ''}".lower()

    if any(k in text for k in ["matrix", "2d array", "grid", "rows", "columns", "transpose"]):
        return "matrix"
    if any(k in text for k in ["palindrome", "anagram", "substring", "string", "character", "vowel", "reverse string"]):
        return "string"
    if any(k in text for k in ["recursion", "recursive", "backtrack", "tower of hanoi", "permutation", "combination"]):
        return "recursion"
    if any(k in text for k in ["dynamic programming", "dp", "fibonacci", "knapsack", "subsequence", "memoization"]):
        return "dp"
    if any(k in text for k in ["binary search", "search", "sort", "bubble sort", "quick sort", "merge sort"]):
        return "sorting_searching"
    if any(k in text for k in ["linked list", "node", "next pointer", "doubly linked"]):
        return "linked_list"
    if any(k in text for k in ["pointer", "malloc", "calloc", "dynamic memory", "dereference"]):
        return "pointers"
    if any(k in text for k in ["structure", "struct", "record", "union"]):
        return "structures"
    if any(k in text for k in ["prime", "gcd", "lcm", "modulo", "factorial", "digit", "sum of digits", "binary", "hexadecimal"]):
        return "math_number"
    if any(k in text for k in ["stack", "queue", "parentheses", "fifo", "lifo"]):
        return "stack_queue"
    return "general_array_loop"


def _generate_diagnostic_hint(submission, hint_number):
    """Legacy wrapper for submission-based diagnostic generation."""
    return _generate_diagnostic_hint_for_question(
        submission.question if submission else None,
        hint_number,
        submission=submission
    )


def _generate_diagnostic_hint_for_question(question, hint_number, submission=None):
    """
    Topic-aware, question-grounded theory engine for when LLM is unavailable or offline.
    Directly references the question title and domain theory with ZERO CODE.
    """
    if not question:
        return (
            "🧠 Theoretical Foundations: Understand the fundamental problem constraints, definitions, and invariant properties.\n"
            "🎯 Conceptual Boundary Principles: Analyze edge cases including empty domains, identity elements, and single inputs."
        )

    topic = _detect_question_topic(question)
    title = question.title
    status = submission.status if submission else None
    error_output = (submission.error_output or "").lower() if submission else ""
    failed_test = (
        _extract_failed_test_info(getattr(submission, "judge_output", None))
        if submission and status == Submission.Status.WRONG_ANSWER
        else None
    )

    # 1. Compilation Errors (Grammar, Lexical Scope, and Type Theory)
    if status == Submission.Status.COMPILE_ERROR and submission:
        if "expected ';'" in error_output or "expected statement before" in error_output:
            return (
                f"🧠 Formal Grammar & Parsing Theory ({title}): Programming languages are governed by formal context-free grammars. The compiler parser requires syntactic statements to strictly adhere to language production rules with proper statement delimiters.\n"
                f"🎯 Conceptual Correction: Review the grammatical structure of your theoretical statements. Ensure each conceptual statement is terminated completely before initiating the next."
            )
        if "undeclared" in error_output or "undefined" in error_output:
            return (
                f"🧠 Lexical Scope & Identifier Binding ({title}): Identifiers must be formally bound to an active lexical scope before they can be referenced in computational expressions. Unbound references indicate an out-of-scope symbol.\n"
                f"🎯 Conceptual Correction: Verify that the theoretical entity is declared within the active block scope or parameter environment before evaluating it."
            )
        if "too few arguments" in error_output or "too many arguments" in error_output:
            return (
                f"🧠 Type Theory & Function Signatures ({title}): Function calls require exact structural conformance with the declared formal parameter signature, matching arity and domain types.\n"
                f"🎯 Conceptual Correction: Verify that the arity (number of arguments) and expected conceptual domain types match the theoretical signature."
            )
        return (
            f"🧠 Formal Grammar Parsing ({title}): The parser encountered an unparsed token violating language grammar productions.\n"
            f"🎯 Conceptual Correction: Focus exclusively on the very first syntactic production error to resolve cascading compiler diagnostics."
        )

    # 2. Time Limit Exceeded (Computational Complexity & Asymptotic Growth)
    if status == Submission.Status.TLE:
        if hint_number == 1:
            return (
                f"🧠 Computational Complexity & Asymptotic Growth ({title}): The problem constraints require an algorithm with lower asymptotic time complexity. High-order polynomial or exponential growth fails when input scales.\n"
                f"🎯 Conceptual Boundary Principles: Verify that iterative and state processes strictly advance toward termination on every state transition."
            )
        elif hint_number == 2:
            return (
                f"⚙️ Algorithmic Paradigm & Complexity Reduction ({title}): A quadratic process can often be reduced to linear or linearithmic time by utilizing spatial indexing, divide-and-conquer partitioning, or frequency tables.\n"
                f"🛠️ Conceptual Logic Flow: Eliminate redundant inner scans by precomputing intermediate properties or maintaining monotonic properties across elements."
            )
        else:
            return (
                f"🔍 Redundant Subproblems & Memoization Theory ({title}): Repeating identical subproblem evaluations creates combinatorial explosion in execution time.\n"
                f"⚠️ Conceptual Pitfalls: Recognize overlapping subproblems and apply state caching or iterative bottom-up tabulation to evaluate each unique subproblem at most once."
            )

    # 3. Runtime Errors (Memory Isolation & Arithmetic Singularities)
    if status == Submission.Status.RUNTIME_ERROR:
        if "segmentation fault" in error_output or "sigsegv" in error_output or "core dumped" in error_output:
            return (
                f"🧠 Memory Addressing & Virtual Memory Architecture ({title}): Accessing memory locations outside the allocated virtual address space violates memory isolation invariants, triggering system segmentation faults.\n"
                f"🎯 Conceptual Boundary Principles: Ensure index evaluations remain strictly within the half-open interval from zero up to the cardinal size of the collection."
            )
        if "floating point exception" in error_output or "division by zero" in error_output:
            return (
                f"🧠 Arithmetic Safety Theory ({title}): In arithmetic theory, division by zero is undefined and represents an arithmetic singularity that triggers an instantaneous hardware exception.\n"
                f"🎯 Conceptual Boundary Principles: Guarantee that the divisor operand strictly excludes zero under all possible input permutations before performing modular or division operations."
            )

    # 4. Logical Errors (Theory-Grounded by Question Topic & Title)
    topic_guides = {
        "matrix": {
            1: (
                f"🧠 Discrete Coordinate Mapping & Grid Geometry ({title}): A matrix is a two-dimensional collection of elements indexed by discrete spatial coordinates (row, column). In geometric transformations such as transposition, each coordinate (r, c) reflects across the principal identity diagonal to (c, r).\n"
                f"🎯 Conceptual Boundary Principles: Treat row and column cardinalities independently; algorithms must remain correct for rectangular grids where the number of rows differs from the number of columns."
            ),
            2: (
                f"⚙️ Linear Transformation & Vector Inner Products ({title}): Matrix multiplication is theoretically defined as the linear combination of vector dot products, where each target coordinate (r, c) is the inner product of row vector r from the first matrix and column vector c from the second matrix.\n"
                f"🛠️ Conceptual Logic Flow: Establish three coordinate dimensions: the outer row sweep, the outer column sweep, and the contracting inner dimension that reduces intermediate products into a single scalar."
            ),
            3: (
                f"🔍 Coordinate Invariants & Accumulator Identity ({title}): When calculating cell-by-cell reductions, the scalar accumulator must reset to the additive algebraic identity (zero) prior to evaluating each new coordinate pair.\n"
                f"⚠️ Conceptual Pitfalls: Verify that dimensional limits prevent coordinate overflow beyond the boundary edges of the grid, especially on ragged or asymmetric dimensions."
            ),
        },
        "string": {
            1: (
                f"🧠 Sequence Theory & Reflectional Symmetry ({title}): A string is an ordered finite sequence of discrete symbols over an alphabet. Properties like palindromes represent reflectional symmetry about a central axis, where the symbol at offset k from the origin must match the symbol at offset k from the terminus.\n"
                f"🎯 Conceptual Boundary Principles: Empty sequences and single-symbol sequences are trivially symmetric. For odd-length sequences, the central median symbol serves as an invariant symmetry anchor."
            ),
            2: (
                f"⚙️ Two-Way Convergence & Frequency Distribution Theory ({title}): Symmetrical properties can be verified in linear time through two-way convergence from opposite terminal boundaries toward the center. For anagrams, the character frequency histograms over the alphabet domain must be identical.\n"
                f"🛠️ Conceptual Logic Flow: Check boundary equivalence symmetrically, advancing inward markers while preserving the invariant that all outer symbol pairs have matched."
            ),
            3: (
                f"🔍 Sequence Delimiters & Equivalence Relations ({title}): Differences in letter case, non-printable control symbols, or whitespace alter sequence length and equivalence.\n"
                f"⚠️ Conceptual Pitfalls: Ensure that sequence evaluations distinguish between semantic content symbols and terminal delimiters, and apply canonical normalization if case-insensitivity is theoretically required."
            ),
        },
        "recursion": {
            1: (
                f"🧠 Principle of Mathematical Induction ({title}): Recursion is computational induction: it relies on proving a base case for minimal inputs and establishing an inductive step where solving problem size K enables solving size K+1.\n"
                f"🎯 Conceptual Boundary Principles: The base case serves as the fundamental axiom halting infinite regression. It must comprehensively cover all minimal domain boundaries such as zero, one, or empty states."
            ),
            2: (
                f"⚙️ Recurrence Relations & Problem Decomposition ({title}): Express the problem mathematically as a recurrence relation that expresses the global solution as a combination of strictly smaller, identical subproblems.\n"
                f"🛠️ Conceptual Logic Flow: Guarantee that every recursive branch strictly reduces the distance metric to the base condition, synthesizing sub-results during the unwind phase."
            ),
            3: (
                f"🔍 Well-Founded Relations & Call Stack Invariants ({title}): Infinite recursion occurs when the reduction step fails along a well-founded ordering, allowing cyclic or non-decreasing arguments.\n"
                f"⚠️ Conceptual Pitfalls: Verify that unexpected edge cases or negative values cannot bypass the base case condition, avoiding unbounded call-stack growth."
            ),
        },
        "dp": {
            1: (
                f"🧠 Bellman's Principle of Optimality ({title}): Dynamic Programming applies when a problem satisfies optimal substructure: an optimal global solution incorporates optimal solutions to its constituent subproblems.\n"
                f"🎯 Conceptual Boundary Principles: Identify the overlapping subproblem space and establish the initial boundary states representing the base values of the recurrence."
            ),
            2: (
                f"⚙️ State Space Representation & Transition Theory ({title}): Formalize each subproblem as a discrete state that encapsulates all historical choices needed to evaluate future decisions.\n"
                f"🛠️ Conceptual Logic Flow: Establish the recurrence transition that computes larger states from prerequisite smaller states, following a topological evaluation order without redundant computation."
            ),
            3: (
                f"🔍 State Dependency Graph & Numerical Invariants ({title}): Incorrect results in dynamic programming occur when state transitions are computed before their prerequisite subproblems are fully finalized.\n"
                f"⚠️ Conceptual Pitfalls: Confirm that the order of subproblem evaluation strictly respects the topological dependency graph, and check for numerical overflow in additive state progressions."
            ),
        },
        "sorting_searching": {
            1: (
                f"🧠 Monotonicity & Binary Partitioning Theory ({title}): Searching an ordered domain relies on monotonicity: a monotonic sequence guarantees that a comparison at the midpoint divides the search domain into two partitions, discarding half the search space.\n"
                f"🎯 Conceptual Boundary Principles: If the sequence is monotonically non-decreasing, elements strictly less than the target cannot reside in the upper partition, halving the domain in logarithmic time."
            ),
            2: (
                f"⚙️ Logarithmic Search Invariants & Divide-and-Conquer ({title}): Maintain the invariant that if the target exists in the sequence, it is strictly enclosed between the active lower and upper interval boundaries.\n"
                f"🛠️ Conceptual Logic Flow: Continually contract the candidate interval toward the target or collapse to an empty domain, proving non-existence in logarithmic O(log N) steps."
            ),
            3: (
                f"🔍 Interval Degeneracy & Convergence Criteria ({title}): Off-by-one errors in binary search occur when interval updates fail to strictly shrink the domain during single-element or two-element intervals.\n"
                f"⚠️ Conceptual Pitfalls: Ensure that boundary adjustments strictly contract the interval at every comparison step to prevent infinite stagnation."
            ),
        },
        "pointers": {
            1: (
                f"🧠 Indirection & Spatial Address Theory ({title}): Pointers represent memory address indirection, decoupling the identifier of a data entity from its physical storage location in memory space.\n"
                f"🎯 Conceptual Boundary Principles: An address reference must be explicitly bound to valid, active storage before dereferencing its contents."
            ),
            2: (
                f"⚙️ Dynamic Memory Lifecycle & Ownership Semantics ({title}): Dynamic allocation creates memory blocks whose lifetime is managed independently of call frame execution scope.\n"
                f"🛠️ Conceptual Logic Flow: Track address ownership rigorously: allocate storage, verify allocation success, process data through indexed offsets, and release memory upon completion."
            ),
            3: (
                f"🔍 Null Reference & Dangling Pointer Invariants ({title}): Memory safety requires the invariant that an address is non-null and currently points to allocated storage before dereferencing.\n"
                f"⚠️ Conceptual Pitfalls: Avoid altering the base reference address during traversal, which prevents memory leaks and ensures safe reclamation."
            ),
        },
        "linked_list": {
            1: (
                f"🧠 Pointer Chaining & Node Graph Theory ({title}): A linked list is a linear sequence of discrete heap-allocated nodes, where each node stores a value and an explicit pointer reference to its successor node.\n"
                f"🎯 Conceptual Boundary Principles: Empty lists (null head) and single-element lists represent foundational boundary conditions that prevent null pointer exceptions."
            ),
            2: (
                f"⚙️ Reference Manipulation & Invariant Preservation ({title}): Structural modifications (such as insertions, reversals, or deletions) require re-linking node pointers while maintaining connectivity to avoid orphaning successor sublists.\n"
                f"🛠️ Conceptual Logic Flow: Maintain multi-pointer state tracking (such as previous, current, and next references) to update directed edges without losing list continuity."
            ),
            3: (
                f"🔍 Link Integrity & Null Termination Invariants ({title}): A finite linear list must terminate with a null successor pointer. Failing to preserve next-node references before reassigning links creates disconnected sub-graphs or memory leaks.\n"
                f"⚠️ Conceptual Pitfalls: Always buffer a reference to the next node before overwriting the link of the current node, and verify that operations on head nodes update the list anchor."
            ),
        },
        "math_number": {
            1: (
                f"🧠 Number Theory & Fundamental Theorem of Arithmetic ({title}): Every integer greater than one is either prime or can be uniquely factored into primes. Properties like divisibility, parity, and digit decomposition follow formal modular congruence rules.\n"
                f"🎯 Conceptual Boundary Principles: Analyze theoretical domain edge cases including zero, negative values, unit elements (one), and prime versus composite definitions."
            ),
            2: (
                f"⚙️ Divisor Symmetry & Positional Radix Decomposition ({title}): Factors of an integer exist in symmetric pairs centered around its square root, allowing exhaustive factor analysis within square root bounds. In positional number systems, radix division and modulo isolate individual digit magnitudes.\n"
                f"🛠️ Conceptual Logic Flow: Isolate digits systematically through modular residue and integer quotient reduction, or evaluate factor pairs symmetrically up to the square root threshold."
            ),
            3: (
                f"🔍 Discrete Domain Limits & Integer Precision Theory ({title}): Multiplicative operations on integers can exceed finite binary representation ranges, causing silent integer overflow.\n"
                f"⚠️ Conceptual Pitfalls: Account for negative signs in modular arithmetic and verify that accumulators can accommodate intermediate magnitude growth."
            ),
        },
        "structures": {
            1: (
                f"🧠 Composite Type Abstraction & Domain Encapsulation ({title}): Composite types group heterogeneous data attributes into a cohesive abstract entity, maintaining logical encapsulation and data integrity.\n"
                f"🎯 Conceptual Boundary Principles: Each composite instance maintains independent state across its member fields, preserving separation of concerns."
            ),
            2: (
                f"⚙️ Value Semantics versus Reference Semantics ({title}): Passing composite entities by reference allows shared state access without copying overhead, while value semantics guarantee isolation through deep copies.\n"
                f"🛠️ Conceptual Logic Flow: Use reference access for efficiency and shared state modifications, or value access when caller immutability is required."
            ),
            3: (
                f"🔍 Member Initialization & Memory Coherence ({title}): Accessing uninitialized composite attributes reads undefined memory state, contaminating subsequent calculations.\n"
                f"⚠️ Conceptual Pitfalls: Guarantee that all member attributes are explicitly initialized to consistent default states prior to evaluating domain operations."
            ),
        },
        "stack_queue": {
            1: (
                f"🧠 Abstract Data Type Disciplines (LIFO vs. FIFO) ({title}): A stack enforces Last-In-First-Out semantics suitable for reversible state tracking and nested parenthesis balancing, whereas a queue enforces First-In-First-Out ordering for fair chronological scheduling.\n"
                f"🎯 Conceptual Boundary Principles: Empty collections cannot be popped; operational invariants require confirming non-emptiness before attempting element extraction."
            ),
            2: (
                f"⚙️ Structural Invariants & Nested Hierarchies ({title}): In balanced syntax matching, the most recent unmatched opening delimiter must correspond directly to the next incoming closing delimiter.\n"
                f"🛠️ Conceptual Logic Flow: Push contextual states when entering a nested scope, and pop to verify symmetric closure, maintaining structural balance throughout."
            ),
            3: (
                f"🔍 Termination Balance & Underflow Invariants ({title}): Complete theoretical balance requires two invariant conditions: zero underflows during processing, and an empty collection upon reaching sequence termination.\n"
                f"⚠️ Conceptual Pitfalls: Ensure handling covers sequences with unmatched opening symbols left at termination, as well as sequences that begin with premature closing symbols."
            ),
        },
        "general_array_loop": {
            1: (
                f"🧠 Sequence Invariants & Iterative Proof Theory ({title}): Correctness in iterative algorithms relies on loop invariants: theoretical assertions that hold true before iteration begins, remain true across each step, and prove overall problem correctness upon termination.\n"
                f"🎯 Conceptual Boundary Principles: Clearly define the problem domain boundaries, accounting for empty sequences, single-element inputs, and exact start and end coordinates."
            ),
            2: (
                f"⚙️ State Accumulation & Invariant Maintenance ({title}): In cumulative algorithms, an accumulator holds the true aggregated value for the prefix of elements examined so far, starting from the algebraic identity element.\n"
                f"🛠️ Conceptual Logic Flow: Process elements systematically in sequential order, updating state strictly once per element without omission or duplicate processing."
            ),
            3: (
                f"🔍 Fencepost Invariants & Boundary Verification ({title}): Computational discrepancies often arise from fencepost errors: processing one element too few or one element beyond the valid domain.\n"
                f"⚠️ Conceptual Pitfalls: Verify that accumulator states initialize to the true algebraic identity and that termination criteria cover the entire valid domain exactly once."
            ),
        },
    }

    topic_dict = topic_guides.get(topic, topic_guides["general_array_loop"])
    hint = topic_dict.get(hint_number, topic_dict[1])
    if failed_test and "Hidden test case" not in failed_test and "mismatch" not in failed_test:
        hint += f"\n🧪 Your first failing test: {failed_test}. Trace your logic against this exact input by hand before changing code."
    return hint
