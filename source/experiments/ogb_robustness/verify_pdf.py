"""Check every newly generated table value appears in the published PDF."""
from pathlib import Path
import csv,subprocess
H=Path(__file__).resolve().parent;R=H.parents[1]
t=subprocess.check_output(['pdftotext','-layout',str(R/'docs/manuscript/main.pdf'),'-'],text=True)
assert 'Independent ogbn-arxiv streams' in t
rows=list(csv.reader((H/'results/table.csv').open()))
for row in rows[1:]:
 for cell in row:assert cell in t,cell
assert 'Robustness across longer OGB streams' in t
assert 'Numerical verification on longer OGB streams' in t
print('PASS: all robustness table cells and new analysis headings found in PDF')
