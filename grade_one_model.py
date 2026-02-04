# -*- coding: utf-8 -*-
# grade_one_model.py
from __future__ import annotations

import json
from grader.config import (
    OLLAMA_URL, MODEL, INPUT_TEX, BUILD_DIR,
    COMPILE_SCRIPT, CLEAN_PDF, GRADE_JSON, PROMPTS_TXT,
    RAW_MODEL_TXT, RAW_MODEL_EXTRACTED, GRADED_TEX, GRADED_PDF,
    OLLAMA_TIMEOUT,
)
from grader.subprocess_utils import run
from grader.prompts_log import write_prompts_txt
from grader.scoring import compute_total
from grader.ollama_client import call_ollama_grade
from grader.report_builder import build_report_tex, compile_tex_to_pdf
from grader.schema_and_prompts import GRADE_SCHEMA, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE



def main():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_TEX.exists():
        raise FileNotFoundError(f"Missing {INPUT_TEX.resolve()}")

    print(f"Step 1: Compiling student submission with {COMPILE_SCRIPT} ...")
    run(["python", COMPILE_SCRIPT])

    if not CLEAN_PDF.exists():
        raise RuntimeError(f"Expected {CLEAN_PDF} was not created.")

    print(f"Step 2: Grading with model: {MODEL} ...")
    latex_text = INPUT_TEX.read_text(encoding="utf-8", errors="ignore")
    user_prompt = USER_PROMPT_TEMPLATE.format(latex=latex_text)

    write_prompts_txt(
        out_path=PROMPTS_TXT,
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        input_tex_path=INPUT_TEX,
        latex_text=latex_text,
        schema=GRADE_SCHEMA,
    )

    # ✅ grading: robust JSON extraction
    grade = call_ollama_grade(
        ollama_url=OLLAMA_URL,
        model=MODEL,
        system=SYSTEM_PROMPT,
        prompt=user_prompt,
        schema=GRADE_SCHEMA,
        timeout_seconds=OLLAMA_TIMEOUT,
        raw_debug_path=RAW_MODEL_TXT,
        extracted_debug_path=RAW_MODEL_EXTRACTED,
    )

    GRADE_JSON.write_text(json.dumps(grade, ensure_ascii=False, indent=2), encoding="utf-8")

    total = compute_total(grade.get("items", []))

    print("Step 3: Building graded_report.tex ...")
    report_tex = build_report_tex(MODEL, grade, total, PROMPTS_TXT)
    GRADED_TEX.write_text(report_tex, encoding="utf-8")

    print("Step 4: Compiling graded_report.pdf ...")
    compile_tex_to_pdf(GRADED_TEX, BUILD_DIR)


    print("\nDone.")
    print(f"- Grading JSON     : {GRADE_JSON}")
    print(f"- Raw model output : {RAW_MODEL_TXT}")
    print(f"- Extracted JSON   : {RAW_MODEL_EXTRACTED}")
    print(f"- Final graded PDF : {GRADED_PDF}")


if __name__ == "__main__":
    main()
