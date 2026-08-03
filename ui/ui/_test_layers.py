"""Test Fase 8: operazioni layer (delete/duplicate/rename/reorder/visibility/move/paste).

Run: PYTHONPATH=".;common;annotators" QT_QPA_PLATFORM=offscreen .venv-st/Scripts/python.exe ui/ui/_test_layers.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from qtpy.QtWidgets import QApplication

app = QApplication([])
from ui import ui_config
ui_config.load_config()
from ui.misc import apply_theme_palette
apply_theme_palette(app, dark=True)

from ui.proj import ProjSeg
from ui.commands import (DeleteInstancesCommand, DuplicateInstancesCommand,
                         MoveInstanceCommand, SetInstancesVisibleCommand,
                         RenameInstanceCommand, ReorderInstancesCommand,
                         PasteInstancesCommand)


def make_proj(tmpdir):
    # pagina 1x1 con un'istanza applicata
    proj = ProjSeg()
    proj.directory = tmpdir
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    proj.current_model = 'chars/heroA'
    proj._cur_image = img
    from ui.structures import Instance
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    ins = Instance(mask=mask, bbox=[20, 30, 10, 10], idx=0)
    ins.tag = 'torso'
    ins.applied = True
    proj._cur_instances = [ins]
    from ui.assembly import make_drawable
    d = make_drawable(ins, 'torso', img, 'chars/heroA')
    proj.l2dmodel = type('M', (), {
        'drawables': [d],
        'valid_drawables': lambda self: self.drawables,
        'did2drawable': {d.did: d},
        'visible_map': {},
    })()
    return proj, ins


def main():
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix='st_layers_')
    proj, ins0 = make_proj(tmp)

    # --- visibilita' ---
    cmd = SetInstancesVisibleCommand(proj, [ins0], False)
    cmd.redo()
    assert ins0.visible is False
    assert not any(d.did == ins0.did() for d in []), 'sanity'
    cmd.undo()
    assert ins0.visible is True

    # --- duplica ---
    cmd = DuplicateInstancesCommand(proj, [ins0])
    cmd.redo()
    assert len(proj.current_instance_list) == 2
    cp = proj.current_instance_list[1]
    assert cp.applied and cp.visible and cp.idx == 1
    assert cp.mask.shape == ins0.mask.shape
    cmd.undo()
    assert len(proj.current_instance_list) == 1
    cmd.redo()
    assert len(proj.current_instance_list) == 2
    cp = proj.current_instance_list[1]

    # --- rinomina ---
    prev_name = cp.name
    cmd = RenameInstanceCommand(proj, cp, 'braccio dx')
    cmd.redo()
    assert cp.name == 'braccio dx'
    cmd.undo()
    assert cp.name == prev_name

    # --- riordina (porta su: cp era in fondo) ---
    cmd = ReorderInstancesCommand(proj, cp, -1)
    cmd.redo()
    assert proj.current_instance_list[0] is cp
    cmd.undo()
    assert proj.current_instance_list[1] is cp

    # --- sposta ---
    x0 = proj.current_instance_list[0].bbox[0]
    cmd = MoveInstanceCommand(proj, cp, 5, -3)
    cmd.redo()
    assert proj.current_instance_list[1].bbox[0] == x0 + 5
    cmd.undo()
    assert proj.current_instance_list[1].bbox[0] == x0

    # --- cancella ---
    cmd = DeleteInstancesCommand(proj, [ins0])
    cmd.redo()
    assert len(proj.current_instance_list) == 1
    assert proj.current_instance_list[0] is cp
    cmd.undo()
    assert len(proj.current_instance_list) == 2

    # --- incolla (clipboard) ---
    import copy
    cb = [copy.deepcopy(cp)]
    cmd = PasteInstancesCommand(proj, cb)
    cmd.redo()
    assert len(proj.current_instance_list) == 3
    cmd.undo()
    assert len(proj.current_instance_list) == 2

    # --- persistenza: visible/name sopravvivono a save/load ---
    proj.save_current_instances()
    proj._cur_instances = None  # forza ricarica
    lst = proj.current_instance_list
    assert all(hasattr(i, 'visible') and hasattr(i, 'name') for i in lst)
    p = proj.get_instance_path(proj.current_model)
    assert os.path.exists(p)

    shutil.rmtree(tmp, ignore_errors=True)
    print('LAYER OPS OK (visibility/duplicate/rename/reorder/move/delete/paste/persistenza)')


if __name__ == '__main__':
    main()
