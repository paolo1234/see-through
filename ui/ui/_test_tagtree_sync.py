# -*- coding: utf-8 -*-
"""Regressione: dopo Apply as parts la tag tree DEVE conoscere le nuove did
(no KeyError alla selezione canvas), e la cancellazione di un candidato
applicato rimuove anche la drawable. Riepilogo: sample_project -> SAM batch.
Run: PYTHONPATH=".;common;annotators" .venv-st/Scripts/python.exe ui/ui/_test_tagtree_sync.py
"""
import os
import sys
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

mw.on_run_requested(True)
t0 = time.time()
while time.time() - t0 < 240:
    app.processEvents()
    time.sleep(0.05)
    if 'completato' in mw.topArea.status_label.text().lower():
        break
assert 'completato' in mw.topArea.status_label.text().lower()

# Apply as parts (riproduce il click utente)
mw.apply_candidates()
app.processEvents()

drawables = proj.l2dmodel.valid_drawables()
assert len(drawables) > 0, 'nessuna drawable applicata'
dids = {d.did for d in drawables}
known = set(mw.tagtree.did2elem.keys())
print('drawables:', len(dids), '| tagtree conosce:', len(known))
assert dids <= known, ('tagtree non sincronizzata:', dids - known)
print('sync OK: tutte le did applicate sono nella tag tree')

# selezione canvas (il flusso che prima dava KeyError)
mw.canvas.update_drawable_selection(list(dids)[0], True)
app.processEvents()
mw.canvas.update_drawable_selection(list(dids)[0], False)
app.processEvents()
print('selezione canvas OK (nessun KeyError)')

# cancellazione candidato applicato -> drawable rimossa
applied = [i for i in proj.current_instance_list if i.applied]
if applied:
    idx = applied[0].idx
    before = {d.did for d in proj.l2dmodel.valid_drawables()}
    mw.on_candidate_delete(idx)
    app.processEvents()
    after = {d.did for d in proj.l2dmodel.valid_drawables()}
    assert f'inst://{proj.current_model}/{idx}' not in after, 'drawable non rimossa'
    assert after < before, 'nessuna rimozione'
    # tagtree risincronizzata anche dopo la cancellazione
    assert after <= set(mw.tagtree.did2elem.keys())
    print('delete applied OK: drawable %s rimossa' % f'inst://{proj.current_model}/{idx}')

print('TAGTREE SYNC OK')
