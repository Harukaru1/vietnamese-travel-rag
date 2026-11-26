"""Convert a '# %%' cell script into a Jupyter notebook (no jupytext needed).

    python scripts/py_to_ipynb.py kaggle/qlora_grounded_qa.py

Markdown cells start with '# %% [markdown]' and their lines with '# '.
Lines '# !cmd' in code cells become shell commands '!cmd'.
"""

import json
import sys
from pathlib import Path


def convert(src: Path) -> dict:
    cells, kind, lines = [], None, []

    def flush():
        text = "\n".join(lines).strip("\n")
        if kind and text:
            source = [line + "\n" for line in text.split("\n")]
            source[-1] = source[-1].rstrip("\n")
            cell = {"cell_type": kind, "metadata": {}, "source": source}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)

    for line in src.read_text(encoding="utf-8").split("\n"):
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind == "markdown":
            lines.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
        else:
            lines.append(line[2:] if line.startswith("# !") else line)
    flush()
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = src.with_suffix(".ipynb")
    out.write_text(json.dumps(convert(src), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out)
