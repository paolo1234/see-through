"""Test end-to-end: sample_project -> MainWindow -> SAM tiny box -> apply -> export."""
import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from qtpy.QtWidgets import QApplication, QFileDialog, QInputDialog
from qtpy.QtCore import QRectF

app = QApplication([])

from ui.ui import ui_config
ui_config.load_config()
pcfg = ui_config.pcfg
pcfg.inference_provider = 'sam'
pcfg.sam_model_size = 'tiny'
pcfg.segmentation_device = 'cpu'

from ui.ui.mainwindow import MainWindow
from ui.ui.proj import ProjSeg

proj = ProjSeg()
# stato pulito: rimuovi eventuali instances.json di run precedenti
for page in ('chars/heroA', 'chars/heroB'):
    ip = os.path.join('ui/sample_project', page, 'instances.json')
    if os.path.exists(ip):
        os.remove(ip)
proj.load_from_txt('ui/sample_project/proj.txt')
proj.set_current_page_byidx(0)

mw = MainWindow(app, pcfg)
mw.proj = proj
mw.canvas.proj = proj
mw.refresh_candidates()
print('pagine:', len(proj.pages), '| candidati iniziali:', mw.candidates_list.count())

# --- box prompt su 'heroA' (region faccia) via lo stesso path del tool W ---
pcfg.edit_mode = ui_config.EditMode.RectInference
t0 = time.time()
mw.on_end_create_rect(QRectF(120, 40, 272, 160), 0)
t1 = time.time()
print(f'box prompt sam-tiny: {t1-t0:.1f}s')
# processa la coda (thread)
deadline = time.time() + 60
while mw.candidates_list.count() == 0 and time.time() < deadline:
    app.processEvents()
    time.sleep(0.05)
print('candidati dopo box:', mw.candidates_list.count())
assert mw.candidates_list.count() >= 1
print('riga candidato:', mw.candidates_list.item(0).text())

# --- apply come parte (voting su modello senza parti -> unknown) ---
mw.apply_candidates()
app.processEvents()
print('drawable creati:', len(mw.proj.l2dmodel.valid_drawables()))
assert len(mw.proj.l2dmodel.valid_drawables()) >= 1

# --- export layered (monkeypatch dialog) ---
out = os.path.join(mw.proj.instance_dir(), 'layered')
os.makedirs(out, exist_ok=True)
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: out)
mw.export_layered()
files = sorted(os.listdir(out))
print('export files:', files)
assert any(f.endswith('.png') for f in files) and 'manifest.json' in files

print('E2E SAM -> APPLY -> EXPORT OK')
