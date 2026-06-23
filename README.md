# MathGrade

MathGrade is a small Python backend + HTML frontend project for collecting student solutions, generating PDFs, and producing AI-assisted grading feedback.

This README focuses on **the grading pipeline** (how inputs flow through extraction → AI → feedback PDF).

http://localhost:5173/

---

## What the system is trying to do

Instead of asking the model to “read” math from a rendered PDF (which often destroys structure), the system grades from **structured source content**:

- The **reference question + reference solution** (source-of-truth)
- The **student answer** (preferably the original LaTeX snippet per question/part)

The AI produces **feedback in Hebrew** (and optional scores), which is rendered into a readable feedback PDF and merged into the final output PDF.

---

## Pipeline overview

### Inputs

- **Reference**: an official exam/reference PDF that contains the question text and the official solution
- **Student submission**:
  - Recommended: **LaTeX** (`.tex`)
  - Optional: **handwritten** (scanned PDF/images) via OCR

### Outputs

- `payloads/*.json` (or a manifest + payload directory) — per-question/part “grading payloads”
- `grades.json` — AI grading results (structured JSON)
- `graded_feedback.tex` — LaTeX document containing the feedback
- `graded_feedback.pdf` — compiled feedback PDF (XeLaTeX)
- `graded_test.pdf` — final merged PDF (bundle + feedback)

---

## A) LaTeX submission flow (recommended)

### Step A1 — Extract / align per-question content

The system splits content so each AI call matches the correct question/part:

- `reference.question_text` (if available)
- `reference.solution_text`
- `student.latex_raw` (the student’s original LaTeX snippet for that same question/part)

This alignment is important: the model grades **Q1(a) vs Q1(a)**, not the entire exam at once.

### Step A2 — Build rich JSON payloads

For each question/part, the backend generates a JSON payload that is readable for humans and stable for machines. Example:

```json
{
  "question_id": "Q3(b)",
  "reference": {
    "question_text": "...",
    "solution_text": "..."
  },
  "student": {
    "latex_raw": "..."
  },
  "rubric": {
    "score_max": 15,
    "key_points": []
  }
}
```

These payloads are saved on disk so you can debug and reproduce grading.

### Step A3 — AI grading using a local TXT prompt

The grader loads a Hebrew-only grading prompt from a text file located next to the grading code:

- `grading_prompt_he_math.txt`

For each question/part:
- **System message** = contents of `grading_prompt_he_math.txt`
- **User message** = the JSON payload for that question/part

The model must return **valid JSON only** (Option B “rich feedback” format), including:

- `summary_he`
- `feedback_he[]` (actionable feedback items)
- `mismatch` object:
  - `is_mismatch`
  - `reference_target`
  - `student_target`
  - `explanation_he`
- `common_errors_detected[]` (tags)
- `suggested_next_step_he`

Mismatch detection is a key feature:
> If the student solved a different integral/series/claim than the question asks, the AI must say so explicitly.

### Step A4 — Convert AI JSON → LaTeX feedback

After grading all questions/parts, the backend writes:

- `grades.json`
- `graded_feedback.tex`

The LaTeX rendering is designed for readability:

- Hebrew is the default language (RTL)
- Plain text is escaped safely
- Math segments (e.g. `$...$`, `$$...$$`) are preserved so LaTeX commands like `\epsilon`, `\frac`, `\int` render correctly

### Step A5 — Compile feedback LaTeX → PDF

`graded_feedback.tex` is compiled with **XeLaTeX** into `graded_feedback.pdf`.

### Step A6 — Merge PDFs into final output

Finally the system produces a single PDF by merging:

- the bundle PDF (questions + student answers)
- the feedback PDF

Result: `graded_test.pdf` containing the full submission context plus readable feedback.

---

## B) Handwritten submission flow (OCR)

If students submit handwritten answers (scanned PDF/images), you do **not** need to convert everything into LaTeX first.

Recommended approach:

1. Run OCR on the handwriting
2. Produce per-question JSON directly, for example:
   - `student.ocr_text_raw` (raw OCR output)
   - optional `student.math_latex` (if you use math-aware OCR that outputs LaTeX)
   - `student.ocr_confidence` and `uncertain_tokens` (if available)
3. Send this JSON payload into the **same** AI grading step

Why JSON-first works better for handwriting:

- OCR → perfect LaTeX is hard and can introduce silent math errors
- Keeping OCR raw text + confidence makes uncertainty visible
- The model can provide more cautious feedback when OCR is unreliable

---

## Notes on feedback quality

- The system prioritizes **helpful feedback** over scores.
- The model is forced to respond in **Hebrew** and in **valid JSON** (no extra text).
- Mismatch detection prevents “grading the wrong problem” when a student solves a different integral/series/question.
- If the reference PDF is scanned (images, not selectable text), extracting the reference solution may require OCR.

---

## Where to edit the prompt

To change feedback style/tone/rules, edit:

- `flows/student_grading/grading/grading_prompt_he_math.txt`

This file is loaded at runtime from the same directory as the grading module (`prompting.py`), so you do not need to redeploy code to refine feedback instructions—only update the TXT.

---

## Project layout

The code is organized **by user journey**, with cross-cutting code concentrated
in `common/` and platform/web routers in `web/`.

```
app.py                       # FastAPI entrypoint (uvicorn app:app); wires all routers
core/                        # Shared infrastructure
  config.py security.py storage.py debug.py
  ai_clients/                # Ollama / Google / GPT / Mathpix OCR clients + usage logging
schemas/                     # Shared Pydantic models (OCR + reference bundles)
common/                      # Code shared across 2+ journeys
  exam_structure.py
  tex/                       # part_normalize, reference_ranges, reference_tex,
                             # student_tex, compile_tex_to_pdf, clean_tex,
                             # ai_proofreader, latex_render, math_normalize
  pdf/                       # pdf_cleanse
flows/
  student_grading/           # Journey 1: submit .tex -> bundle -> grade -> feedback PDF
    routes.py                #   POST /routes/grade_tex_{ollama,google,chatgpt}
    bundler.py answer_render.py feedback_tex.py unified_tex.py
    grading/                 #   AI grading internals
      payloads.py grader_payloads.py grader.py grader_sources.py
      prompting.py grading_prompt_he_math.txt schema.py
      json_sanitizer.py solution_bank_matcher.py
  solution_bank/             # Journey 2: teacher uploads reference exam -> OCR -> bundles
    routes.py                #   /routes/bank/*
    reference_builder.py
    answers_ocr.py questions_ocr.py questions_answers_ocr.py
    full_solution_service.py
  handwritten_ocr/           # Journey 3: student handwriting -> OCR -> student .tex template
    routes.py                #   POST /routes/ocr_handwritten
    student_work_ocr.py
web/                         # Cross-cutting platform routers
  auth.py stats.py progress.py template_pages.py
  health.py error_handlers.py student_log_routes.py
templates/ static/           # HTML frontend + assets
```

---

## Grading pipeline (LaTeX flow) — module map

The current pipeline is **payload-first**: each question/part is extracted once
into a JSON payload that is the single source of truth for both the bundle PDF
and the AI grading.

```mermaid
flowchart TD
  A[flows/student_grading/routes.py<br/>POST /routes/grade_tex_*] --> B[grading/payloads.py<br/>build_payloads(reference_tex, student_tex)]
  B --> B1[(manifest.json + payloads/*.json<br/>per-question reference + student.latex_raw)]

  %% Bundle TeX is built from the SAME payloads (no re-extraction)
  B1 --> C[bundler.py<br/>_write_bundle_tex_inline_answers()]
  C --> C1[(qa_bundle.tex)]

  %% AI grading drives off the same manifest
  B1 --> D[grading/grader_payloads.py<br/>grade_payload_manifest(manifest, model)]
  D --> D1[grading/prompting.py<br/>load_grading_prompt()]
  D --> D2[core/ai_clients<br/>Ollama / Google / GPT chat_json(schema)]
  D --> D3[(grades.json)]

  %% Feedback TeX, then unify with the bundle, then compile
  D3 --> E[feedback_tex.py<br/>build_feedback_tex(grades.json)]
  E --> E1[(feedback.tex)]
  C1 --> F[unified_tex.py<br/>build_unified_tex(qa_bundle.tex, feedback.tex)]
  E1 --> F
  F --> F1[(graded_provider.tex)]
  F1 --> G[common/tex/compile_tex_to_pdf.py<br/>compile_tex_to_pdf(clean=True, XeLaTeX)]
  G --> G1[(graded_provider.pdf — final output)]
```

