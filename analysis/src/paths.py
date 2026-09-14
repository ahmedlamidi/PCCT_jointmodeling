"""Shared paths. Every script resolves locations from this file, so the
analysis/ folder can be moved as long as it stays beside PcTK_3.24a/."""
import os
SRC   = os.path.dirname(os.path.abspath(__file__))
ANAL  = os.path.dirname(SRC)                     # .../analysis
ROOT  = os.path.dirname(ANAL)                    # .../PCCT_jointmodeling
PCTK  = os.path.join(ROOT, 'PcTK_3.24a')
CACHE = os.path.join(ANAL, 'cache')
FIGS  = os.path.join(ANAL, 'figures')
OUT   = os.path.join(ANAL, 'outputs')
MATLAB= os.path.join(ANAL, 'matlab')
for d in (CACHE, FIGS, OUT): os.makedirs(d, exist_ok=True)
