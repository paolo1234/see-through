# -*- coding: utf-8 -*-
# One-shot installer per TUTTE le dipendenze della UI desktop.
# IMPORTANTE: va eseguito con un Python 3.11/3.12 (es. il venv .venv-st).
# Python 3.13/3.14 NON va bene: numpy/torch non hanno wheel e la build fallisce.
# Se manca almeno un core, installa l'intero requirements-ui-core.txt
# (editable common/annotators + Qt6 + sci/vision stack). Richiamato da launch_studio_ui.bat.
#
# Usa `uv` se presente (veloce, risolve le wheel e NON compila da sorgente),
# altrimenti ripiega su pip (con Python 3.11 le wheel esistono, quindi ok).
import importlib.util, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/ui
REPO = os.path.dirname(HERE)                                 # <repo>
REQ = os.path.join(HERE, "requirements-ui-core.txt")

# moduli di controllo: se manca uno, faccio il full install una tantum
SENTINEL = ["qtpy", "PyQt6", "cv2", "numpy", "PIL", "pycocotools", "win32api"]

def have(mod):
    return importlib.util.find_spec(mod) is not None

missing = [m for m in SENTINEL if not have(m)]

if missing:
    print("[install] dipendenze UI mancanti:", ", ".join(missing))
    print("[install] installo tutto da:", REQ)
    ed_common = os.path.join(REPO, "common")
    ed_annot  = os.path.join(REPO, "annotators")
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable,
               "-e", ed_common, "-e", ed_annot, "-r", REQ]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-U",
               "-e", ed_common, "-e", ed_annot, "-r", REQ]
    r = subprocess.run(cmd)
    sys.exit(r.returncode)

print("[ok] dipendenze UI installate")
sys.exit(0)