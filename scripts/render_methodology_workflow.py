"""Render the methodology-workflow figure as a 600-DPI PNG.

The manuscript embeds this figure as a raster PNG (no inline/vector TikZ in the
submission PDF). This script writes the standalone TikZ source
`fig_methodology_workflow.tex`, compiles it with pdflatex, and rasterizes the
first page to `fig_methodology_workflow.png` at 600 DPI.

Style: grayscale, sharp corners, no shadows; the two phase labels are placed on
the left of each phase box, rotated to read bottom-to-top, on two lines
("Phase I" / "Data and leakage control", and "Phase II" / "Audit, stress
testing, diagnostics"); Phase II contains the Determinism Audit, Ablation,
Stress Tests, and Uncertainty Layer boxes. Fonts match the manuscript
(Fira Sans).

Rasterization tries PyMuPDF (fitz); if unavailable, it falls back to the
Poppler `pdftoppm -r 600` command-line tool.
"""

import os
import shutil
import subprocess

DPI = 600

tikz_code = r"""\documentclass[border=2mm]{standalone}
\usepackage[T1]{fontenc}
\usepackage[scaled=0.90]{FiraSans}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, fit, backgrounds}
\begin{document}
\begin{tikzpicture}[
    auto,
    block/.style={rectangle, draw=black, thick, fill=white, text width=3.3cm, align=center, minimum height=1.6cm, inner sep=2mm, font=\small},
    phase/.style={rectangle, draw=black!50, dashed, thin, inner sep=4mm},
    line/.style={draw, thick, -{Latex[length=3mm,width=2mm]}, color=black},
    plabel/.style={font=\sffamily, text=black, align=center, rotate=90}
]

% Phase 1 Nodes
\node [block] (logs) {\textbf{1. Traffic-Log Export} \\ \vspace{1mm} \footnotesize 1,048,576 records \\ \footnotesize Palo Alto TRAFFIC};
\node [block, right=0.45cm of logs] (target) {\textbf{2. Target Semantics} \\ \vspace{1mm} \footnotesize Allow, Drop, Deny \\ \footnotesize Operational decision};
\node [block, fill=gray!15, right=0.45cm of target] (leakage) {\textbf{3. Proxy Control} \\ \vspace{1mm} \footnotesize Blocked direct fields \\ \footnotesize Policy fields excluded};

% Phase 2 Nodes
\node [block, below=1.5cm of logs] (audit) {\textbf{4. Determinism Audit} \\ \vspace{1mm} \footnotesize Information limits \\ \footnotesize Minimum-key census};
\node [block, fill=gray!15, right=0.45cm of audit] (ablation) {\textbf{5. Ablation Views} \\ \vspace{1mm} \footnotesize No application context \\ \footnotesize Transport+volume only};
\node [block, fill=gray!15, right=0.45cm of ablation] (stress) {\textbf{6. Stress Tests} \\ \vspace{1mm} \footnotesize Temporal split \\ \footnotesize Context-held-out};
\node [block, right=0.45cm of stress] (metrics) {\textbf{7. Uncertainty Layer} \\ \vspace{1mm} \footnotesize Conformal prediction sets \\ \footnotesize Selective risk trade-offs};

% Phase 3 Node
\node [block, fill=gray!25, below=1.5cm of stress, text width=10cm] (boundary) {\textbf{8. Claim Boundary \& Inference} \\ \vspace{1.5mm} \textit{\small Near-deterministic proxy reconstruction; strictly not autonomous attack detection.}};

% Connecting Lines
\path [line] (logs) -- (target);
\path [line] (target) -- (leakage);
\path [line] (leakage.south) -- ++(0,-0.75) -| (audit.north);
\path [line] (audit) -- (ablation);
\path [line] (ablation) -- (stress);
\path [line] (stress) -- (metrics);
\path [line, dashed] (stress.south) -- (boundary.north);

% Background Phases
\begin{scope}[on background layer]
    \node [phase, fit=(logs) (target) (leakage)] (phase1) {};
    \node [phase, fit=(audit) (ablation) (stress) (metrics)] (phase2) {};
\end{scope}

% Vertical two-line phase labels on the left, reading bottom-to-top
\node [plabel] at ([xshift=-7mm]phase1.west) {\textbf{Phase I}\\[1pt] \scriptsize Data and leakage control};
\node [plabel] at ([xshift=-7mm]phase2.west) {\textbf{Phase II}\\[1pt] \scriptsize Audit, stress testing, diagnostics};

\end{tikzpicture}
\end{document}
"""


def rasterize(pdf_path, png_path, dpi):
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(png_path)
        return "PyMuPDF"
    except Exception:
        if shutil.which("pdftoppm") is None:
            raise RuntimeError("Neither PyMuPDF nor pdftoppm is available for PNG conversion.")
        stem = png_path[:-4] if png_path.lower().endswith(".png") else png_path
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", pdf_path, stem], check=True)
        return "pdftoppm"


def main():
    tex_path = "fig_methodology_workflow.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tikz_code)

    print("Compiling TikZ to PDF...")
    res = subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path], capture_output=True)
    if res.returncode != 0 or not os.path.exists("fig_methodology_workflow.pdf"):
        print("LaTeX compilation failed!")
        return

    print(f"Rasterizing to {DPI} DPI PNG...")
    backend = rasterize("fig_methodology_workflow.pdf", "fig_methodology_workflow.png", DPI)
    print(f"Done via {backend}: fig_methodology_workflow.png")


if __name__ == "__main__":
    main()
