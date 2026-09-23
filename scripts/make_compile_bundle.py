# -*- coding: utf-8 -*-
"""Build a self-contained LaTeX compile bundle for the manuscript.

Collects cas-sc-sample.tex, every ACTIVE \\includegraphics asset (comments
stripped), and any class/bibliography/style files found next to the tex, into a
single zip that can be uploaded to Overleaf (or unpacked anywhere) and compiled
without missing-file errors.
"""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXDIR = ROOT / 'manuscript'
TEX = TEXDIR / 'cas-sc-sample.tex'
OUT = ROOT / 'LSC_XAI_compile_bundle.zip'


def uncomment(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)


text = TEX.read_text(encoding='utf-8')
assets = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', uncomment(text))

extra_exts = {'.cls', '.sty', '.bib', '.bst', '.bbl', '.clo', '.cfg', '.tex'}
extras = [p for p in TEXDIR.iterdir()
          if p.is_file() and p.suffix.lower() in extra_exts and p != TEX]

included, skipped = [], []
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(TEX, TEX.name)
    included.append(TEX.name)
    for rel in sorted(set(assets)):
        src = TEXDIR / rel
        if src.exists():
            zf.write(src, rel)
            included.append(rel)
        else:
            skipped.append(rel)
    for p in sorted(extras):
        zf.write(p, p.name)
        included.append(p.name)

print(f'bundle: {OUT.name} ({OUT.stat().st_size / 1e6:.2f} MB)')
print(f'included {len(included)} files: {TEX.name}, {len(set(assets)) - len(skipped)} figures, {len(extras)} class/bib/style files')
if skipped:
    print('SKIPPED (referenced but absent, all in comments?):', skipped)
