"""Export the executed notebook: uv run python scripts/export_part3_pdf.py.

Install the notebook dependency group and Tectonic before running this script.
"""

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import nbformat
from nbconvert import LatexExporter
from nbconvert.writers import FilesWriter
import pypandoc

ROOT = Path(__file__).resolve().parents[1]
NAME = "project_notebook_part3"

# Pandoc's bundled binary is notebook-only and need not be installed system-wide.
os.environ["PATH"] = str(Path(pypandoc.get_pandoc_path()).parent) + os.pathsep + os.environ["PATH"]
notebook = nbformat.read(ROOT / f"{NAME}.ipynb", as_version=4)
for cell in notebook.cells:
    if cell.cell_type == "code" and (
        cell.execution_count is None
        or any(output.output_type == "error" for output in cell.outputs)
    ):
        raise ValueError("Run every notebook cell successfully before exporting.")

# nbconvert preserves code, output text, and plots. Wrap long code/output lines
# so the printed version remains readable within the page margins.
latex, resources = LatexExporter().from_notebook_node(
    notebook, resources={
        "metadata": {"name": "IND320 - Part 3: Data quality"},
        "output_files_dir": f"{NAME}_files",
    }
)
latex = latex.replace(
    r"\usepackage{fancyvrb}",
    "\\usepackage{fancyvrb}\n\\usepackage{fvextra}\n\\fvset{breaklines=true,breakanywhere=true}",
)
with TemporaryDirectory(prefix="ind320-pdf-") as folder:
    FilesWriter(build_directory=folder).write(latex, resources, notebook_name=NAME)
    subprocess.run(
        ["tectonic", f"{NAME}.tex", "--outdir", str(ROOT)],
        cwd=folder, check=True,
    )
print(f"Exported {ROOT / f'{NAME}.pdf'}")
