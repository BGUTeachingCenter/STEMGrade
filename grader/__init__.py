# grader/__init__.py
"""
grader package

Public API:
- compile_tex_file: compile a .tex to cleaned .tex + optional PDF/DOCX
- generate_qa_bundle_pdf: produce a clean "bundle" PDF that embeds reference + student pages
"""

from .compile_tex import compile_tex_file, CompileResult
from .qa_bundle import generate_qa_bundle_pdf, BundleOutputs

__all__ = [
    "compile_tex_file",
    "CompileResult",
    "generate_qa_bundle_pdf",
    "BundleOutputs",
]
