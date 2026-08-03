"""Regression test Fase 5: mask_ops + comandi undo-able (split/merge/edit)."""
import os

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QUndoStack

app = QApplication([])

from ui.ui.structures import Instance
from ui.ui.mask_ops import paint, split_components, merge_masks, bbox_of, morphology
from ui.ui.commands import (EditMaskCommand, SplitInstanceCommand,
                            MergeInstancesCommand)

H, W = 120, 160


def make_ins(mask, bbox, idx, tag='unknown', applied=False):
    ins = Instance(mask=mask, bbox=bbox, score=0.9, idx=idx)
    ins.tag = tag
    ins.applied = applied
    return ins


# ---- paint ----
m = np.zeros((30, 40), bool)
m2 = paint(m, [(10, 10)], [1], radius=5)
assert m2.sum() > 0 and m.sum() == 0  # originale non mutato
m3 = paint(m2, [(10, 10)], [0], radius=10)
assert m3.sum() < m2.sum()
print('paint OK')

# ---- split ----
m = np.zeros((60, 60), bool)
m[10:25, 10:25] = True
m[40:55, 40:55] = True
pieces = split_components(m)
assert len(pieces) == 2, len(pieces)
print('split_components OK:', [p.shape for p in pieces])

# ---- merge ----
a = make_ins(np.ones((15, 15), bool), [10, 10, 15, 15], 0)
b = make_ins(np.ones((15, 15), bool), [40, 40, 15, 15], 1)
mm, bb = merge_masks([a.mask, b.mask], [a.bbox, b.bbox], H, W)
assert bb == [10, 10, 45, 45], bb
assert mm.shape == (45, 45)
print('merge_masks OK:', bb)

# ---- proj reale (API completa) -------
import tempfile
import os
import numpy as np
from PIL import Image

from ui.ui.proj import ProjSeg

class FakeLModel:
    drawables = []

    def valid_drawables(self):
        return [d for d in self.drawables if d.area > 0]


tmp = tempfile.mkdtemp(prefix='st_f5_')
page = 'models/charA'
os.makedirs(os.path.join(tmp, page), exist_ok=True)
Image.fromarray(np.full((120, 160, 3), 200, np.uint8)).save(os.path.join(tmp, page, 'final.png'))

proj = ProjSeg()
proj.directory = tmp
proj.pages = {page: []}
proj._pagename2idx = {page: 0}
proj._idx2pagename = {0: page}
proj.current_model = page
proj.proj_path = os.path.join(tmp, 'test.json')
proj._cur_instances = [a, b]
proj.l2dmodel = FakeLModel()

stack = QUndoStack()

# ---- EditMaskCommand ----
ins_e = make_ins(np.ones((20, 20), bool), [50, 50, 20, 20], 5)
proj._cur_instances.append(ins_e)
new_m = np.zeros((20, 20), bool)
new_m[5:15, 5:15] = True
stack.push(EditMaskCommand(proj, ins_e, new_m, [55, 55, 10, 10]))
assert ins_e.bbox == [55, 55, 10, 10] and ins_e.mask.shape == (20, 20)
stack.undo()
assert ins_e.bbox == [50, 50, 20, 20]
stack.redo()
assert ins_e.bbox == [55, 55, 10, 10]
print('EditMaskCommand undo/redo OK')

# ---- SplitInstanceCommand ----
big = make_ins(m, [0, 0, 60, 60], 9)
proj._cur_instances.append(big)
stack.push(SplitInstanceCommand(proj, big, split_components(m)))
assert big not in proj._cur_instances
assert len(proj._cur_instances) == 5  # a, b, ins_e + 2 parti
stack.undo()
assert big in proj._cur_instances and len(proj._cur_instances) == 4
stack.redo()
assert big not in proj._cur_instances
print('SplitInstanceCommand undo/redo OK')

# ---- MergeInstancesCommand ----
stack.push(MergeInstancesCommand(proj, [a, b], mm, bb))
assert a not in proj._cur_instances and b not in proj._cur_instances
assert len(proj._cur_instances) == 4  # ins_e + 2 parti + merged
merged = [i for i in proj._cur_instances if i.idx not in {0, 1, 9} and i.bbox == bb]
assert len(merged) == 1
stack.undo()
assert a in proj._cur_instances and b in proj._cur_instances
assert len(proj._cur_instances) == 5
stack.redo()
assert a not in proj._cur_instances
print('MergeInstancesCommand undo/redo OK')

# ---- morphologia ----
mm2 = morphology(np.ones((20, 20), bool), 'erode', ksize=5)
assert mm2.sum() < 400 and mm2.sum() > 0
print('morphology OK')

print('FASE 5 COMPLETA')
