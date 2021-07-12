#!/usr/bin/env python3

import os
import select
import sys

import nbformat


def _cells(nb):
    """Yield all cells in an nbformat-insensitive manner"""
    if nb.nbformat < 4:
        for ws in nb.worksheets:
            for cell in ws.cells:
                yield cell
    else:
        for cell in nb.cells:
            yield cell


def strip_output(nb):
    """strip the outputs from a notebook object"""
    nb.metadata.pop('signature', None)
    for cell in _cells(nb):
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
        if 'prompt_number' in cell:
            cell['prompt_number'] = None
    return nb


def remove_widget_state(nb):
    if 'widgets' in nb['metadata']:
        del nb['metadata']['widgets']
    return nb


if __name__ == "__main__":
    nb_bytes = b''
    
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if sys.stdin in ready:
            data = os.read(sys.stdin.fileno(), 4096)
            if len(data) == 0:
                break
            nb_bytes += data
    
    nb_string = nb_bytes.decode("utf-8")
    nb = nbformat.reads(nb_string, as_version=nbformat.NO_CONVERT)
    
    nb = strip_output(nb)
    nb = remove_widget_state(nb)
    
    nbformat.write(nb, sys.stdout.fileno())

