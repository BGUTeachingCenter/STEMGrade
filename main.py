# main.py
from pathlib import Path
from grader.qa_bundle import generate_qa_bundle_pdf

def main():
    outputs = generate_qa_bundle_pdf(
        reference_pdf=Path("midtermCalc2sol.pdf"),
        student_tex=Path("012.tex"),
        out_dir=Path("build"),
        bundle_stem="qa_bundle",
        font_name="Arial",
    )
    print("Bundle PDF:", outputs.bundle_pdf)
    print("Bundle TEX:", outputs.bundle_tex)
    print("Student PDF:", outputs.student_clean_pdf)
    print("Ref ranges:", outputs.ref_ranges)
    print("Student ranges:", outputs.student_ranges)

if __name__ == "__main__":
    main()
