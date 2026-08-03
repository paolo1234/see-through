# -*- coding: utf-8 -*-
"""Test export layers PNG + PSD (Fase 9)."""
import os, sys, tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from qtpy.QtWidgets import QApplication
app = QApplication([])
from ui import ui_config
ui_config.load_config()
from ui.misc import apply_theme_palette
apply_theme_palette(app, dark=True)
import numpy as np
from ui.proj import ProjSeg
from ui.structures import Instance
from ui.assembly import make_drawable
from ui.export.layers import collect_parts, export_layers_png, export_layers_psd

tmp = tempfile.mkdtemp(prefix='st_layx_')
img = np.zeros((100, 100, 3), np.uint8); img[..., 1] = 120
proj = ProjSeg()
proj.directory = tmp
proj.current_model = 'heroA'
proj._cur_image = img

# drawable reali via assembly.make_drawable (mask CROP-LOCAL)
mask1 = np.zeros((30, 30), np.uint8); mask1[:, :] = 255
ins1 = Instance(mask=mask1, bbox=[10, 10, 30, 30], idx=0, score=0.9)
ins1.tag = 'head'; ins1.applied = True
d1 = make_drawable(ins1, 'head', img, 'heroA')
mask2 = np.zeros((40, 60), np.uint8); mask2[:, :] = 255
ins2 = Instance(mask=mask2, bbox=[10, 50, 60, 40], idx=1, score=0.8)
ins2.tag = 'torso'; ins2.applied = True
d2 = make_drawable(ins2, 'torso', img, 'heroA')
proj.l2dmodel = type('M', (), {
    'drawables': [d1, d2],
    'valid_drawables': lambda self: self.drawables,
    'did2drawable': {d1.did: d1, d2.did: d2},
    'visible_map': {},
})()
proj._cur_instances = [ins1, ins2]

parts = collect_parts(proj)
assert len(parts) == 2, f'parti attese 2, trovate {len(parts)}'
assert all(p['img'].shape[2] == 4 for p in parts)
assert sorted(p['tag'] for p in parts) == ['head', 'torso']

out_dir = os.path.join(tmp, 'layers_out')
files = export_layers_png(proj, parts, out_dir)
assert len(files) == 4, f'attesi 4 file, trovati {len(files)}'
assert any(f.endswith('background.png') for f in files)
assert any(f.endswith('preview.png') for f in files)
assert any('head' in os.path.basename(f) for f in files)
assert any('torso' in os.path.basename(f) for f in files)

psd_path = os.path.join(tmp, 'layers.psd')
n = export_layers_psd(proj, parts, psd_path)
assert n == 2, f'layer PSD attesi 2, scritti {n}'
assert os.path.exists(psd_path) and os.path.getsize(psd_path) > 1000

from psd_tools import PSDImage
psd = PSDImage.open(psd_path)
names = [lyr.name for lyr in psd]
assert 'background' in names and any('head' in x for x in names), names
print(f'LAYERS EXPORT OK: {len(files)} PNG + PSD {psd.size} con {len(psd)} layer')
