import os
import sys

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QUndoStack

app = QApplication([])

from live2d.scrap_model import Drawable
from ui.ui.assembly import vote_tags
from ui.ui.structures import Instance
from ui.ui.commands import CreateDrawablesCommand

H, W = 120, 160
img = np.full((H, W, 3), 200, np.uint8)


def make_drawable_stub(tag, xyxy, mask_shape):
    d = Drawable(img=np.zeros((mask_shape[1], mask_shape[0], 4), np.uint8),
                 crop_xyxy=list(xyxy), pad_drawable_img=False,
                 seg_type='body_part_tag', src_size=(H, W))
    d.x, d.y = int(xyxy[0]), int(xyxy[1])
    d.set_tag(tag)
    d.visible_mask = np.ones((mask_shape[1], mask_shape[0]), bool)
    d.area = int(mask_shape[0] * mask_shape[1])
    return d


lm_drawables = [
    make_drawable_stub('hair', [20, 30, 80, 70], (70, 80)),
    make_drawable_stub('face', [30, 90, 60, 40], (40, 60)),
]


class FakeLModel:
    drawables = lm_drawables

    def valid_drawables(self):
        return [d for d in self.drawables if d.area > 0]


lm = FakeLModel()

# istanze: mask CROP-LOCALE (bbox-relative), come producono i provider
ins1 = Instance(mask=np.ones((50, 70), bool), bbox=[25, 35, 70, 50], score=0.9, idx=0)
ins2 = Instance(mask=np.ones((30, 50), bool), bbox=[35, 95, 50, 30], score=0.8, idx=1)
ins3 = Instance(mask=np.ones((10, 10), bool), bbox=[5, 5, 10, 10], score=0.4, idx=2)

votes = vote_tags(lm, [ins1, ins2, ins3], img, min_iou=0.15)
tags = [t for _, t, _, _ in votes]
print('voting ->', tags)
assert tags == ['hair', 'face', 'unknown'], tags
print('VOTING OK')

page = 'models/charA'
proj = type('P', (), {})()
proj.l2dmodel = lm
proj.current_model = page
proj._cur_instances = [ins1, ins2, ins3]
proj.save_current_instances = lambda: None

stack = QUndoStack()
cmd = CreateDrawablesCommand(proj, [ins1, ins2, ins3], tags, page, img)
stack.push(cmd)
assert len(lm.drawables) == 5, len(lm.drawables)
assert all(i.applied for i in [ins1, ins2, ins3])
new_tags = [d.tag for d in lm.drawables[2:]]
print('redo: drawables =', len(lm.drawables), '; tag nuovi:', new_tags)
assert new_tags == ['hair', 'face', 'unknown']
d = lm.drawables[2]
print('drewable[2]: did=%s img=%s vis=%s x,y=%s,%s' % (
    d.did, d.img.shape, d.visible_mask.shape, d.x, d.y))

stack.undo()
assert len(lm.drawables) == 2 and not ins1.applied
print('undo OK')

stack.redo()
assert len(lm.drawables) == 5
print('redo OK  ->  FASE 3 COMPLETA')
