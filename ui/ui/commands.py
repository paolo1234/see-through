from typing import List, Callable, Dict

from qtpy.QtCore import QPointF
try:
    from qtpy.QtWidgets import QUndoStack, QUndoCommand
except:
    from qtpy.QtGui import QUndoStack, QUndoCommand

from .drawable_item import DrawableItem
from .proj import ProjSeg


class SetDrawableTagCommand(QUndoCommand):
    def __init__(self, update_tag: Callable, src_tags, tgt_tags):
        super(SetDrawableTagCommand, self).__init__()
        self.src_tags = src_tags
        self.tgt_tags = tgt_tags
        self.update_tag = update_tag

    def redo(self):
        self.update_tag(self.tgt_tags)

    def undo(self):
        self.update_tag(self.src_tags)



class AddInstancesCommand(QUndoCommand):
    """Undo-able aggiunta di istanze candidato alla pagina corrente."""
    def __init__(self, proj, instances, persist=True):
        super().__init__()
        self.proj = proj
        self.instances = list(instances)
        self.persist = persist
        self.text()

    def redo(self):
        cur = self.proj.current_instance_list
        ids = {i.idx for i in cur}
        for ins in self.instances:
            if ins.idx in ids:
                ins.idx = max(ids, default=-1) + 1
                ids.add(ins.idx)
            cur.append(ins)
        if self.persist:
            self.proj.save_current_instances()

    def undo(self):
        cur = self.proj.current_instance_list
        rm_ids = {i.idx for i in self.instances}
        self.proj._cur_instances = [i for i in cur if i.idx not in rm_ids]
        if self.persist:
            self.proj.save_current_instances()


class CommonCommand(QUndoCommand):
    def __init__(self, redo_kwargs, undo_kwargs, func: Callable):
        super().__init__()
        self.redo_kwargs = redo_kwargs
        self.undo_kwargs = undo_kwargs
        self.func = func

    def redo(self):
        self.func(**self.redo_kwargs)

    def undo(self):
        self.func(**self.undo_kwargs)