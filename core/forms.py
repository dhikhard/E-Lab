from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.template.defaultfilters import slugify
from typing import ClassVar

from .models import Course, Module, OpenEndedQuestion, Question, Quiz, Submission, TestCase, User, SubQuestion


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput(attrs={"accept": ".csv,.txt,.pdf", "multiple": True}))
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_clean = super().clean
        if isinstance(data, (list, tuple)):
            return [single_clean(file, initial) for file in data]
        return [single_clean(data, initial)]


SEMESTER_CHOICES = [(n, f"Semester {n}") for n in range(1, 9)]


class StudentSignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)
    department = forms.ChoiceField(label="Branch", choices=[("", "Choose your branch")] + list(User.Branch.choices))
    semester = forms.TypedChoiceField(choices=SEMESTER_CHOICES, coerce=int, initial=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].label = "USN"
        self.fields["username"].help_text = "Use your student USN, for example NNM25CC001."

    class Meta:
        model = User
        fields = ("username", "first_name", "last_name", "email", "department", "semester")

    def save(self, commit=True):
        user = super().save(commit=False)
        user.role = User.Role.STUDENT
        user.year = (user.semester + 1) // 2
        if not user.usn:
            user.usn = user.username
        if commit:
            user.save()
        return user


class SubmissionForm(forms.ModelForm):
    subquestion = forms.ModelChoiceField(
        queryset=SubQuestion.objects.none(),  # Will be set in __init__
        required=False,
        widget=forms.HiddenInput()
    )

    class Meta:
        model = Submission
        fields = ("code", "subquestion")
        widgets: ClassVar[dict] = {
            "code": forms.Textarea(attrs={"rows": 25, "spellcheck": "false", "class": "code-editor"}),
        }

    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        question = kwargs.pop('question', None)
        subquestion_id = kwargs.pop('subquestion_id', None)
        super().__init__(*args, **kwargs)
        
        # If we have a question, set the subquestion queryset to its subquestions
        if question:
            self.fields['subquestion'].queryset = question.subquestions.filter(is_active=True)
            # If we have a specific subquestion_id, set it as initial
            if subquestion_id:
                try:
                    subquestion = question.subquestions.get(id=subquestion_id, is_active=True)
                    self.fields['subquestion'].initial = subquestion
                except SubQuestion.DoesNotExist:
                    pass


class ModuleForm(forms.ModelForm):
    class Meta:
        model = Module
        fields = ("name", "description", "level", "order", "is_active")


class CSVQuestionUploadForm(forms.Form):
    files = MultipleFileField(
        label="Question files",
        help_text="Upload CSV, TXT, or PDF files. CSV files use column headers; TXT/PDF files use structured === QUESTION === blocks.",
    )


QUESTION_LANGUAGE_CHOICES = [
    (50, "C (GCC)"),
    (54, "C++ (G++)"),
    (62, "Java 17"),
    (71, "Python 3"),
    (82, "SQL (DBMS modules)"),
]


class QuestionForm(forms.ModelForm):
    language_id = forms.TypedChoiceField(choices=QUESTION_LANGUAGE_CHOICES, coerce=int, initial=50, label="Language")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["module"].empty_label = "Choose a module"
        self.fields["module"].queryset = Module.objects.select_related("course").order_by("course__name", "order", "name")
        self.fields["title"].widget.attrs.update({"placeholder": "Longest substring without repeating characters"})
        self.fields["slug"].required = False
        self.fields["slug"].help_text = "Leave blank to build it from the title."
        self.fields["slug"].widget.attrs.update({"placeholder": "longest-substring"})
        self.fields["csv_level"].label = "Level"
        self.fields["csv_level"].help_text = "Order within the difficulty (1 = first)."
        self.fields["description"].widget.attrs.update(
            {
                "placeholder": (
                    "Describe the task, input format, output format, and constraints. "
                    "Students should read from stdin and print only the expected answer."
                )
            }
        )
        self.fields["sample_input"].widget.attrs.update({"placeholder": "abcabcbb"})
        self.fields["sample_output"].widget.attrs.update({"placeholder": "3"})
        self.fields["starter_code"].widget.attrs.update(
            {
                "placeholder": (
                    "#include <stdio.h>\n\n"
                    "int main(void) {\n"
                    "    /* Read from stdin and print exact output only. */\n"
                    "    return 0;\n"
                    "}\n"
                )
            }
        )
        self.fields["time_limit"].help_text = "Seconds allowed for each test case."
        self.fields["memory_limit_kb"].help_text = "Memory limit in KB. 128000 is usually enough for C basics."

    def clean(self):
        cleaned = super().clean()
        module = cleaned.get("module")
        slug = slugify(cleaned.get("slug") or cleaned.get("title") or "")[:170] or "question"
        if module:
            taken = Question.objects.filter(module=module)
            if self.instance.pk:
                taken = taken.exclude(pk=self.instance.pk)
            base, counter = slug, 1
            while taken.filter(slug=slug).exists():
                slug, counter = f"{base}-{counter}", counter + 1
        cleaned["slug"] = slug
        return cleaned

    class Meta:
        model = Question
        fields = (
            "module",
            "title",
            "slug",
            "description",
            "difficulty",
            "csv_level",
            "sample_input",
            "sample_output",
            "starter_code",
            "language_id",
            "allow_multiple_languages",
            "time_limit",
            "memory_limit_kb",
            "is_mandatory",
            "proctoring_enabled",
            "is_active",
        )
        widgets: ClassVar[dict] = {
            "description": forms.Textarea(attrs={"rows": 8}),
            "starter_code": forms.Textarea(attrs={"rows": 15, "class": "code-editor"}),
            "sample_input": forms.Textarea(attrs={"rows": 4}),
            "sample_output": forms.Textarea(attrs={"rows": 4}),
            "csv_level": forms.NumberInput(attrs={"min": 1}),
        }


class TestCaseForm(forms.ModelForm):
    class Meta:
        model = TestCase
        fields = ("question", "stdin", "expected_output", "is_sample", "order")
        widgets: ClassVar[dict] = {
            "stdin": forms.Textarea(attrs={"rows": 4}),
            "expected_output": forms.Textarea(attrs={"rows": 4}),
        }


class QuickTestCaseForm(forms.ModelForm):
    class Meta:
        model = TestCase
        fields = ("stdin", "expected_output", "is_sample", "order")
        widgets: ClassVar[dict] = {
            "stdin": forms.Textarea(attrs={"rows": 4, "placeholder": "Input passed through stdin"}),
            "expected_output": forms.Textarea(attrs={"rows": 4, "placeholder": "Exact expected output"}),
            "order": forms.NumberInput(attrs={"min": 1}),
        }


class ProfileForm(forms.ModelForm):
    """Form for users to edit their own profile details."""

    class Meta:
        model = User
        fields = ("first_name", "last_name", "email", "bio")
        widgets: ClassVar[dict] = {
            "first_name": forms.TextInput(attrs={"placeholder": "First name"}),
            "last_name": forms.TextInput(attrs={"placeholder": "Last name"}),
            "email": forms.EmailInput(attrs={"placeholder": "you@example.com"}),
            "bio": forms.Textarea(attrs={"rows": 3, "placeholder": "A short bio about yourself…", "maxlength": 500}),
        }


class FacultyCourseSelectForm(forms.Form):
    """Form for faculty to select which courses they manage."""

    courses = forms.ModelMultipleChoiceField(
        queryset=Course.objects.filter(is_active=True),
        widget=forms.CheckboxSelectMultiple,
        required=False,
        label="Select courses to manage",
        help_text="You will only see students, questions, and progress for your selected courses.",
    )


class OpenEndedQuestionForm(forms.ModelForm):
    class Meta:
        model = OpenEndedQuestion
        fields = ("course", "title", "description", "assigned_date", "due_date", "is_active")
        widgets: ClassVar[dict] = {
            "description": forms.Textarea(attrs={"rows": 8}),
            "assigned_date": forms.DateInput(attrs={"type": "date"}),
            "due_date": forms.DateInput(attrs={"type": "date"}),
        }


class QuizForm(forms.ModelForm):
    class Meta:
        model = Quiz
        fields = ("course", "title", "description", "duration_minutes", "passing_score", "start_time", "end_time", "is_active", "show_results")
        widgets: ClassVar[dict] = {
            "description": forms.Textarea(attrs={"rows": 4}),
            "start_time": forms.DateTimeInput(attrs={"type": "datetime-local"}),
            "end_time": forms.DateTimeInput(attrs={"type": "datetime-local"}),
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields:
            if field != 'description' and field != 'is_active' and field != 'show_results':
                self.fields[field].required = True