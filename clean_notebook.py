#!/usr/bin/env python3
"""git clean filter: strip outputs from a .ipynb read on stdin, write to stdout.

Stdlib only, on purpose. This runs from .git/config on every `git status` in an
archived repo; a filter that needs a third-party import is a filter that breaks
the day its interpreter changes, and a broken clean filter makes every notebook
in the tree look modified by tens of thousands of output lines.
"""

import json
import sys


def _cells(nb):
    """Yield all cells, nbformat-3 worksheets included."""
    if nb.get("nbformat", 4) < 4:
        for ws in nb.get("worksheets", []):
            for cell in ws.get("cells", []):
                yield cell
    else:
        for cell in nb.get("cells", []):
            yield cell


def strip_output(nb):
    nb.get("metadata", {}).pop("signature", None)
    for cell in _cells(nb):
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
        if "prompt_number" in cell:
            cell["prompt_number"] = None
        if "metadata" in cell:
            jupyter = cell["metadata"].get("jupyter")
            cell["metadata"] = {"jupyter": jupyter} if jupyter is not None else {}
    return nb


def remove_widget_state(nb):
    nb.get("metadata", {}).pop("widgets", None)
    return nb


if __name__ == "__main__":
    nb = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    nb = remove_widget_state(strip_output(nb))
    # nbformat.write's exact serialisation, so the filter is a no-op on files
    # already committed through the nbformat-based version of this script.
    sys.stdout.write(json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False))
    sys.stdout.write("\n")
