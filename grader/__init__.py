# grader/__init__.py
"""
Modular grading package.

Key modules:
- subprocess_utils: run() helper for subprocess with nice errors
- prompts_log: writes prompts + hashes log
- scoring: compute_total()
- schema_and_prompts: GRADE_SCHEMA + prompts
- json_safety: robust JSON extraction/repair
- ollama_client: streaming call to Ollama + safe JSON parse
- tex_format: TeX escaping + RTL/LTR safe formatting
- report_builder: build_report_tex() + compile_tex_to_pdf()
"""
