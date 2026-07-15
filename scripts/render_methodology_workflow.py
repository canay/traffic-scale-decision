"""Render the manuscript methodology workflow as a 600-DPI PNG.

The author-controlled TikZ source is compiled with XeLaTeX because the Akis1
figure typography standard uses Lato Bold for headings and Liberation Sans for
supporting text. The script never rewrites the source file. It writes the final
artifact directly to ``figures_diagnostics/fig_methodology_workflow.png`` and
removes successful-build sidecars.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


DPI = 600
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TEX_PATH = SCRIPT_DIR / "fig_methodology_workflow.tex"
PDF_PATH = SCRIPT_DIR / "fig_methodology_workflow.pdf"
PNG_PATH = REPO_ROOT / "figures_diagnostics" / "fig_methodology_workflow.png"


def run_checked(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def rasterize() -> str:
    try:
        import fitz  # PyMuPDF

        with fitz.open(PDF_PATH) as document:
            page = document.load_page(0)
            zoom = DPI / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pixmap.save(PNG_PATH)
        return "PyMuPDF"
    except ImportError:
        pdftoppm = shutil.which("pdftoppm")
        if pdftoppm is None:
            raise RuntimeError("PyMuPDF or pdftoppm is required for rasterization.")
        output_stem = PNG_PATH.with_suffix("")
        run_checked(
            [
                pdftoppm,
                "-png",
                "-r",
                str(DPI),
                "-singlefile",
                str(PDF_PATH),
                str(output_stem),
            ]
        )
        return "pdftoppm"


def clean_sidecars() -> None:
    for suffix in (".aux", ".log", ".out", ".pdf", ".xdv"):
        path = SCRIPT_DIR / f"fig_methodology_workflow{suffix}"
        path.unlink(missing_ok=True)


def main() -> None:
    if shutil.which("xelatex") is None:
        raise RuntimeError("XeLaTeX is required to embed Lato and Liberation Sans.")
    if not TEX_PATH.exists():
        raise FileNotFoundError(TEX_PATH)

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            "xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            TEX_PATH.name,
        ]
    )
    backend = rasterize()
    clean_sidecars()
    print(f"Rendered via {backend}: {PNG_PATH} ({DPI} DPI)")


if __name__ == "__main__":
    main()
