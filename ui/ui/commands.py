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
            self.proj.l2dmodel.did2drawable[d.did] = d
        for ins in self.instances:
            ins.applied = True
        self.proj.save_current_instances()

    def undo(self):
        dids = {did for did, _ in self.added}
        self.proj.l2dmodel.drawables = [
            d for d in self.proj.l2dmodel.drawables if d.did not in dids]
        for did in dids:
            self.proj.l2dmodel.did2drawable.pop(did, None)
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


class EditMaskCommand(QUndoCommand):
    """Undo-able: sostituisce la maschera di un'istanza (brush/eraser/morfologia)."""
    def __init__(self, proj, ins, new_mask, new_bbox):
        super().__init__()
        self.proj = proj
        self.ins = ins
        self.src_mask = ins.mask
        self.src_bbox = ins.bbox
        self.tgt_mask = new_mask
        self.tgt_bbox = new_bbox

    def _apply(self, mask, bbox):
        self.ins.mask = mask
        self.ins.bbox = bbox
        self.ins._contours = None
        self.proj.invalidate_applied_drawable(self.ins)
        self.proj.save_current_instances()

    def redo(self):
        self._apply(self.tgt_mask, self.tgt_bbox)

    def undo(self):
        self._apply(self.src_mask, self.src_bbox)


class SplitInstanceCommand(QUndoCommand):
    """Undo-able: divide un'istanza nelle sue componenti connesse."""
    def __init__(self, proj, ins, pieces, min_area=8):
        super().__init__()
        self.proj = proj
        self.ins = ins
        self.src = (ins.mask, ins.bbox, ins.tag, ins.applied)
        self.min_area = min_area
        self.pieces = list(pieces)
        self.created = []  # nuove istanze (ins, bbox)

    def redo(self):
        if not self.created:
            from .mask_ops import bbox_of
            cur = self.proj.current_instance_list
            ids = {i.idx for i in cur}
            nxt = max(ids, default=-1) + 1
            x, y, _, _ = (int(v) for v in self.ins.bbox)
            for pc in self.pieces:
                bx, by, bw, bh = bbox_of(pc)
                from .structures import Instance
                new_ins = Instance(mask=pc, bbox=[x + bx, y + by, bw, bh],
                                   score=self.ins.score, idx=nxt)
                new_ins.tag = self.ins.tag
                new_ins.applied = self.ins.applied
                self.created.append(new_ins)
                nxt += 1
        # rimuovi l'originale, aggiungi le parti
        cur = self.proj.current_instance_list
        cur[:] = [i for i in cur if i is not self.ins] + self.created
        self.ins.applied = False
        self.proj.invalidate_applied_drawable(self.ins)
        self.proj.save_current_instances()

    def undo(self):
        cur = self.proj.current_instance_list
        rm = {i.idx for i in self.created}
        cur[:] = [i for i in cur if i.idx not in rm] + [self.ins]
        self.ins.mask, self.ins.bbox, self.ins.tag, self.ins.applied = self.src
        self.proj.save_current_instances()


class MergeInstancesCommand(QUndoCommand):
    """Undo-able: fonde piu' istanze in una (amalgamated parts)."""
    def __init__(self, proj, instances, merged_mask, merged_bbox):
        super().__init__()
        self.proj = proj
        self.instances = list(instances)
        self.merged_mask = merged_mask
        self.merged_bbox = merged_bbox
        self.new_ins = None

    def redo(self):
        from .structures import Instance
        cur = self.proj.current_instance_list
        ids = {i.idx for i in cur}
        nxt = max(ids, default=-1) + 1
        src = self.instances[0]
        self.new_ins = Instance(mask=self.merged_mask, bbox=self.merged_bbox,
                                score=max(i.score for i in self.instances), idx=nxt)
        self.new_ins.tag = max((i.tag for i in self.instances), key=lambda t: sum(
            1 for i in self.instances if i.tag == t))
        rm = {id(i) for i in self.instances}
        keep = [i for i in cur if id(i) not in rm]
        cur[:] = keep + [self.new_ins]
        for i in self.instances:
            i.applied = False
            self.proj.invalidate_applied_drawable(i)
        self.proj.save_current_instances()

    def undo(self):
        cur = self.proj.current_instance_list
        cur[:] = [i for i in cur if i is not self.new_ins] + self.instances
        for i in self.instances:
            self.proj.rebuild_applied_drawable(i)
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


class DeleteInstancesCommand(QUndoCommand):
    """Elimina layer/istanze (undoable)."""

    def __init__(self, proj, instances, refresh_cb=None):
        super().__init__('Elimina layer')
        self.proj = proj
        self.instances = list(instances)
        self.refresh_cb = refresh_cb
        lst = proj.current_instance_list
        self._pos = [lst.index(i) for i in self.instances]

    def redo(self):
        self.proj.delete_instances(self.instances)
        if self.refresh_cb:
            self.refresh_cb()

    def undo(self):
        lst = self.proj.current_instance_list
        for pos, ins in sorted(zip(self._pos, self.instances)):
            lst.insert(min(pos, len(lst)), ins)
        for i in self.instances:
            if i.applied and i.visible:
                self.proj.rebuild_applied_drawable(i)
        self.proj.save_current_instances()
        if self.refresh_cb:
            self.refresh_cb()


class DuplicateInstancesCommand(QUndoCommand):
    """Duplica layer (undoable)."""

    def __init__(self, proj, instances, refresh_cb=None):
        super().__init__('Duplica layer')
        self.proj = proj
        self.instances = list(instances)
        self.refresh_cb = refresh_cb
        self._new = []

    def redo(self):
        self._new = self.proj.duplicate_instances(self.instances)
        if self.refresh_cb:
            self.refresh_cb()

    def undo(self):
        self.proj.delete_instances(self._new)
        if self.refresh_cb:
            self.refresh_cb()


class MoveInstanceCommand(QUndoCommand):
    """Sposta un layer di (dx, dy) px (undoable)."""

    def __init__(self, proj, ins, dx, dy, refresh_cb=None):
        super().__init__('Sposta layer')
        self.proj = proj
        self.ins = ins
        self.dx = int(dx)
        self.dy = int(dy)
        self.refresh_cb = refresh_cb

    def redo(self):
        self.proj.move_instance(self.ins, self.dx, self.dy)
        if self.refresh_cb:
            self.refresh_cb()

    def undo(self):
        self.proj.move_instance(self.ins, -self.dx, -self.dy)
        if self.refresh_cb:
            self.refresh_cb()


class SetInstancesVisibleCommand(QUndoCommand):
    """Mostra/nascondi layer (undoable)."""

    def __init__(self, proj, instances, visible, refresh_cb=None):
        super().__init__('Mostra/nascondi layer')
        self.proj = proj
        self.instances = list(instances)
        self.visible = bool(visible)
        self.refresh_cb = refresh_cb

    def redo(self):
        self.proj.set_instances_visible(self.instances, self.visible)
        if self.refresh_cb:
            self.refresh_cb()

    def undo(self):
        self.proj.set_instances_visible(self.instances, not self.visible)
        if self.refresh_cb:
            self.refresh_cb()


class RenameInstanceCommand(QUndoCommand):
    """Rinomina un layer (undoable)."""

    def __init__(self, proj, ins, name, refresh_cb=None):
        super().__init__('Rinomina layer')
        self.proj = proj
        self.ins = ins
        self.name = name
        self._prev = ins.name
        self.refresh_cb = refresh_cb

    def redo(self):
        self.proj.rename_instance(self.ins, self.name)
        if self.refresh_cb:
            self.refresh_cb()

    def undo(self):
        self.proj.rename_instance(self.ins, self._prev)
        if self.refresh_cb:
            self.refresh_cb()


class ReorderInstancesCommand(QUndoCommand):
    """Sposta un layer su/giu nell'ordine di disegno (undoable)."""

    def __init__(self, proj, ins, delta, refresh_cb=None):
        super().__init__('Riordina layer')
        self.proj = proj
        self.ins = ins
        self.delta = int(delta)
        self.refresh_cb = refresh_cb

    def redo(self):
        self.proj.reorder_instance(self.ins, self.delta)
        if self.refresh_cb:
            self.refresh_cb()

    def undo(self):
        self.proj.reorder_instance(self.ins, -self.delta)
        if self.refresh_cb:
            self.refresh_cb()


class PasteInstancesCommand(QUndoCommand):
    """Incolla layer dagli appunti (copie applicate, undoable)."""

    def __init__(self, proj, templates, refresh_cb=None):
        super().__init__('Incolla layer')
        self.proj = proj
        self.templates = list(templates)
        self.refresh_cb = refresh_cb
        self._new = []

    def _make(self):
        import copy
        new = []
        for t in self.templates:
            c = copy.deepcopy(t)
            c.idx = self.proj.next_free_idx()
            c.applied = True
            c.visible = True
            c.name = (t.name or f'inst://{self.proj.current_model}/{t.idx}') + ' (copia)'
            self.proj.current_instance_list.append(c)
            new.append(c)
            self.proj.rebuild_applied_drawable(c)
        self.proj.save_current_instances()
        return new

    def redo(self):
        self._new = self._make()
        if self.refresh_cb:
            self.refresh_cb()

    def undo(self):
        self.proj.delete_instances(self._new)
        if self.refresh_cb:
            self.refresh_cb()
