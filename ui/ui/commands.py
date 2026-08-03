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


class CreateDrawablesCommand(QUndoCommand):
    """Undo-able: promuove istanze candidato a Drawable nel modello di pagina."""
    def __init__(self, proj, instances, tags, page, img):
        super().__init__()
        self.proj = proj
        self.instances = list(instances)
        self.tags = list(tags)
        self.page = page
        self.img = img
        self.added = []  # [(did, drawable)]

    def _build(self):
        from .assembly import make_drawable
        for ins, tag in zip(self.instances, self.tags):
            d = make_drawable(ins, tag, self.img, self.page)
            if d is None:
                continue
            d.draw_order = len(self.proj.l2dmodel.drawables) + len(self.added)
            d.idx = d.draw_order
            self.added.append((d.did, d))

    def redo(self):
        if not self.added:
            self._build()
        for _, d in self.added:
            self.proj.l2dmodel.drawables.append(d)
        for ins in self.instances:
            ins.applied = True
        self.proj.save_current_instances()

    def undo(self):
        dids = {did for did, _ in self.added}
        self.proj.l2dmodel.drawables = [
            d for d in self.proj.l2dmodel.drawables if d.did not in dids]
        for ins in self.instances:
            ins.applied = False
        self.proj.save_current_instances()


class SetCandidateTagsCommand(QUndoCommand):
    """Undo-able: ri-tagga istanze candidato (voting o manuale)."""
    def __init__(self, proj, instances, tags):
        super().__init__()
        self.proj = proj
        self.instances = list(instances)
        self.tags = list(tags)
        self.src_tags = [ins.tag for ins in self.instances]

    def redo(self):
        for ins, tag in zip(self.instances, self.tags):
            ins.tag = tag
        self.proj.save_current_instances()

    def undo(self):
        for ins, tag in zip(self.instances, self.src_tags):
            ins.tag = tag
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