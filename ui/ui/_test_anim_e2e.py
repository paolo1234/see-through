# -*- coding: utf-8 -*-
"""E2E: sample_project -> SAM batch -> Apply -> dialogo animazione -> export atlas.
Run: PYTHONPATH=".;common;annotators" .venv-st/Scripts/python.exe ui/ui/_test_anim_e2e.py
"""
import os
import sys
import tempfile
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication

app = QApplication([])

from ui import ui_config
ui_config.load_config()
pcfg = ui_config.pcfg

from ui.misc import apply_theme_palette
apply_theme_palette(app, dark=True)

from ui.mainwindow import MainWindow
from ui.proj import ProjSeg

proj = ProjSeg()
proj.load('ui/sample_project/proj.txt')
mw = MainWindow(app, pcfg)
mw.proj = proj
mw.canvas.proj = proj
mw.canvas.clearDrawableItems()
mw.updatePageList()
app.processEvents()

pcfg.inference_provider = 'sam'
pcfg.sam_model_size = 'tiny'
pcfg.segmentation_device = 'cpu'

# batch SAM sulla prima pagina (auto-segmentazione)
mw.on_run_requested(True)
t0 = time.time()
while time.time() - t0 < 240:
    app.processEvents()
    time.sleep(0.05)
    if 'completato' in mw.topArea.status_label.text().lower():
        break
assert 'completato' in mw.topArea.status_label.text().lower(), mw.topArea.status_label.text()
print('batch OK: candidati =', mw.candidates_list.count())

# applica i candidati come parti
mw.apply_candidates()
app.processEvents()
drawables = proj.l2dmodel.valid_drawables()
print('parti applicate =', len(drawables))
assert len(drawables) > 0

# dialogo animazione
from ui.anim_dialog import AnimationDialog
dlg = AnimationDialog(proj, mw)
dlg._generate()  # carica parti + genera ciclo dello stato corrente
assert 0 < len(dlg._parts) <= len(drawables), (len(dlg._parts), len(drawables))
cur = dlg.state_combo.currentText()
assert cur == 'walk', cur
assert cur in dlg._cycles and len(dlg._cycles[cur]) == int(dlg.frames_spin.value())
print('genera OK: parti=%d, stato=%s, tag=%s' % (len(dlg._parts), cur, [p['tag'] for p in dlg._parts][:6]))

# preview frame
dlg._render_current()
assert not dlg.preview.pixmap().isNull()
print('preview OK')

# export (bypass dialog: scrive su dir temporanea)
tmp = tempfile.mkdtemp(prefix='st_anim_e2e_')
import qtpy.QtWidgets as QW
QW.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: tmp)
dlg._export()
sheet = os.path.join(tmp, 'sprite-sheet-alpha.png')
assert os.path.isfile(sheet), os.listdir(tmp)
assert os.path.isfile(os.path.join(tmp, 'manifest.json'))
import json
m = json.load(open(os.path.join(tmp, 'manifest.json'), encoding='utf-8'))
assert set(m['rows'].keys()) == {'idle', 'walk', 'run'}, m['rows'].keys()
for st, rects in m['rows'].items():
    for r in rects:
        assert r['w'] == m['cellWidth'] and r['h'] == m['cellHeight']
print('export OK:', sheet, 'cell=%dx%d' % (m['cellWidth'], m['cellHeight']))

print('E2E ANIM OK')
