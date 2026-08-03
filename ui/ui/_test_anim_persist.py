# -*- coding: utf-8 -*-
"""Test persistenza parametri animazione (anim.json, Fase 9)."""
import os, sys, tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from qtpy.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from ui.proj import ProjSeg
from ui.structures import Instance
from ui.assembly import make_drawable
from ui.anim.cycles import CycleParams

tmp = tempfile.mkdtemp(prefix='st_animp_')
img = np.zeros((100, 100, 3), np.uint8); img[..., 1] = 120
proj = ProjSeg()
proj.directory = tmp
proj.current_model = 'heroA'
proj._cur_image = img
mask = np.zeros((30, 30), np.uint8); mask[:, :] = 255
ins = Instance(mask=mask, bbox=[10, 10, 30, 30], idx=0, score=0.9)
ins.tag = 'torso'; ins.applied = True
d = make_drawable(ins, 'torso', img, 'heroA')
proj.l2dmodel = type('M', (), {
    'drawables': [d],
    'valid_drawables': lambda self: self.drawables,
    'did2drawable': {d.did: d},
    'visible_map': {},
})()
proj._cur_instances = [ins]

# 1) serializzazione
p = CycleParams.defaults('run'); p.leg_swing = 55.0; p.frames = 12; p.duration_s = 0.5
dct = p.to_dict()
p2 = CycleParams.from_dict(dct)
assert p2.leg_swing == 55.0 and p2.frames == 12 and p2.duration_s == 0.5
assert p2.kind == 'run'
# from_dict tollerante
p3 = CycleParams.from_dict({'kind': 'bogus', 'leg_swing': 'x'})
assert p3.kind == 'walk' and p3.leg_swing == 30.0  # default walk

# 2) salva/carica via proj
proj.save_anim({'walk': p.to_dict(), 'run': CycleParams.defaults('run').to_dict()})
loaded = proj.load_anim()
assert set(loaded) == {'walk', 'run'}, loaded
assert loaded['walk']['leg_swing'] == 55.0
# percorso
assert os.path.exists(proj.anim_path())
# file corrotto -> {}
with open(proj.anim_path(), 'w') as f:
    f.write('not json{')
assert proj.load_anim() == {}

# 3) dialog carica i saved (via proj.load_anim) e li riscrive a chiusura
proj.save_anim({'walk': p.to_dict()})
from ui.anim_dialog import AnimationDialog
dlg = AnimationDialog(proj)
dlg.state_combo.setCurrentText('walk')
assert dlg.duration_spin.value() == 0.5, dlg.duration_spin.value()
assert dlg.leg_spin.value() == 55.0
# simulazione chiusura (closeEvent salva)
dlg.close()
reloaded = proj.load_anim()
assert 'walk' in reloaded and 'idle' in reloaded and 'run' in reloaded, reloaded
print('ANIM PERSIST OK: anim.json salvato/ricaricato, dialog ripristina i parametri')
