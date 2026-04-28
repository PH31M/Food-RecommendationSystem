import nbformat
import sys
import os

def summarize_nb(path, out_f):
    out_f.write(f'\n========================================\n')
    out_f.write(f'--- Notebook: {path} ---\n')
    out_f.write(f'========================================\n\n')
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown':
            out_f.write(f'\n[Markdown Cell {i}]\n')
            out_f.write(cell.source + '\n')
        elif cell.cell_type == 'code':
            out_f.write(f'\n[Code Cell {i}]\n')
            lines = cell.source.split('\n')
            if len(lines) > 10:
                out_f.write('\n'.join(lines[:10]) + '\n... (truncated)\n')
            else:
                out_f.write(cell.source + '\n')
            if 'outputs' in cell and cell.outputs:
                out_f.write('--- Outputs ---\n')
                for out in cell.outputs:
                    if out.output_type == 'stream':
                        out_text = out.text.split('\n')
                        if len(out_text) > 10:
                            out_f.write('\n'.join(out_text[:10]) + '\n... (truncated)\n')
                        else:
                            out_f.write(out.text + '\n')
                    elif out.output_type in ['execute_result', 'display_data']:
                        if 'text/plain' in out.data:
                            out_text = out.data['text/plain'].split('\n')
                            if len(out_text) > 10:
                                out_f.write('\n'.join(out_text[:10]) + '\n... (truncated)\n')
                            else:
                                out_f.write(out.data['text/plain'] + '\n')

with open('scratch/notebook_summary.txt', 'w', encoding='utf-8') as out_f:
    summarize_nb('notebooks/03_data_feature_engineering.ipynb', out_f)
    summarize_nb('notebooks/04_recommendation_model.ipynb', out_f)
