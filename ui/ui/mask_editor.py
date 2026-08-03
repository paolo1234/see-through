"""Fase 5: editor di maschera per istanza candidato.

Dialog con superficie di disegno (immagine + overlay maschera):
- brush / eraser con raggio regolabile (undo locale Ctrl+Z)
- split in componenti connesse (parti amalgamate)
- erode / dilate / reset
Al confermo (OK) restituisce la/e maschera/e risultanti in coordinate
crop-locali + bbox full-size, pronti per i comandi undo-able di commands.py.
"""
from typing import List, Tuple

import cv2
import numpy as np
from qtpy.QtCore import QPointF, QRectF, QSize, Qt
from qtpy.QtGui import QColor, QImage, QPainter, QPen
from qtpy.QtWidgets import (QButtonGroup, QDialog, QHBoxLayout, QLabel, QPushButton,
                            QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from .mask_ops import bbox_of, morphology, paint, split_components


class _MaskSurface(QWidget):
    """Superficie di disegno: crop dell'immagine + maschera overlay."""

    def __init__(self, page_img: np.ndarray, mask: np.ndarray, bbox, parent=None):
        super().__init__(parent)
        x, y, w, h = (int(v) for v in bbox)
        self.ox, self.oy, self.bw, self.bh = x, y, w, h
        self.crop = page_img[y:y + h, x:x + w].copy()
        self.mask = np.asarray(mask, dtype=bool).copy()
        if self.mask.shape != (h, w):
            self.mask = cv2.resize(self.mask.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST) > 0
        self.tool = 'brush'
        self.radius = 8
        self.undo_stack: List[np.ndarray] = []
        self._last = None
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:
        return QSize(520, 520)

    def _scale(self):
        if self.crop.size == 0:
            return 1.0
        iw, ih = self.crop.shape[1], self.crop.shape[0]
        return min(self.width() / max(iw, 1), self.height() / max(ih, 1))

    def _img_pos(self, pos):
        s = self._scale()
        if s <= 0:
            return None
        iw, ih = self.crop.shape[1], self.crop.shape[0]
        ox = (self.width() - iw * s) / 2
        oy = (self.height() - ih * s) / 2
        return (int((pos.x() - ox) / s), int((pos.y() - oy) / s))

    def _stroke(self, pos, add_undo=True):
        p = self._img_pos(pos)
        if p is None:
            return
        x, y = p
        if not (0 <= x < self.bw and 0 <= y < self.bh):
            return
        lab = 1 if self.tool == 'brush' else 0
        if add_undo and (self._last is None or
                         (abs(self._last[0] - x) + abs(self._last[1] - y) > self.radius)):
            self.undo_stack.append(self.mask.copy())
            if len(self.undo_stack) > 60:
                self.undo_stack.pop(0)
        self._last = (x, y)
        self.mask = paint(self.mask, [(x, y)], [lab], radius=self.radius)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._stroke(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._stroke(event.pos(), add_undo=False)

    def undo(self):
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(40, 40, 40))
        if self.crop.size == 0:
            return
        s = self._scale()
        iw, ih = self.crop.shape[1], self.crop.shape[0]
        ox = (self.width() - iw * s) / 2
        oy = (self.height() - ih * s) / 2
        rgb = self.crop.copy()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        qimg = QImage(rgb.data, iw, ih, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        target = QRectF(ox, oy, iw * s, ih * s)
        p.drawImage(target, qimg)
        # overlay maschera
        mask = np.zeros((ih, iw, 4), dtype=np.uint8)
        mask[self.mask, 0] = 255
        mask[self.mask, 3] = 130
        qm = QImage(mask.data, iw, ih, mask.strides[0], QImage.Format.Format_RGBA8888).copy()
        p.drawImage(target, qm)
        # contorni
        p.setPen(QPen(QColor(255, 255, 255), max(1, int(s))))
        m8 = self.mask.astype(np.uint8)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            path = [QPointF(ox + pt[0][0] * s, oy + pt[0][1] * s) for pt in c]
            if path:
                p.drawPolyline(path)
        p.end()

    def split(self):
        pieces = split_components(self.mask)
        if len(pieces) < 2:
            return pieces
        self.undo_stack.append(self.mask.copy())
        self.mask = pieces[0]  # componente principale nell'editor
        self.update()
        return pieces[1:]

    def morphology_op(self, op: str):
        self.undo_stack.append(self.mask.copy())
        self.mask = morphology(self.mask, op, ksize=self.radius)
        self.update()


class MaskEditorDialog(QDialog):
    """Editor modale per una singola istanza candidato."""

    def __init__(self, page_img: np.ndarray, mask: np.ndarray, bbox,
                 title: str = 'Edit mask', parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(720, 640)

        self.surface = _MaskSurface(page_img, mask, bbox)
        self.extra_pieces: List[np.ndarray] = []
        self.result_pieces: List[Tuple[np.ndarray, List[int]]] = []

        # toolbar
        self.btn_brush = QToolButton(self); self.btn_brush.setText('Brush')
        self.btn_brush.setCheckable(True); self.btn_brush.setChecked(True)
        self.btn_eraser = QToolButton(self); self.btn_eraser.setText('Eraser')
        self.btn_eraser.setCheckable(True)
        grp = QButtonGroup(self)
        grp.addButton(self.btn_brush); grp.addButton(self.btn_eraser)
        grp.buttonClicked.connect(self._on_tool)

        self.radius_spin = QSpinBox(self)
        self.radius_spin.setRange(2, 60); self.radius_spin.setValue(8)
        self.radius_spin.valueChanged.connect(self._on_radius)

        btn_split = QPushButton('Split', self); btn_split.clicked.connect(self._on_split)
        btn_erode = QPushButton('Erode', self); btn_erode.clicked.connect(lambda: self.surface.morphology_op('erode'))
        btn_dilate = QPushButton('Dilate', self); btn_dilate.clicked.connect(lambda: self.surface.morphology_op('dilate'))
        btn_undo = QPushButton('Undo', self); btn_undo.clicked.connect(self.surface.undo)
        btn_reset = QPushButton('Reset', self); btn_reset.clicked.connect(self._on_reset)
        btn_ok = QPushButton('OK', self); btn_ok.clicked.connect(self._on_ok)
        btn_cancel = QPushButton('Cancel', self); btn_cancel.clicked.connect(self.reject)

        bar = QHBoxLayout()
        for w in (self.btn_brush, self.btn_eraser, QLabel(' R:'), self.radius_spin,
                  btn_split, btn_erode, btn_dilate, btn_undo, btn_reset):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(btn_ok)
        bar.addWidget(btn_cancel)

        lay = QVBoxLayout(self)
        lay.addLayout(bar)
        lay.addWidget(self.surface, 1)

        self._info = QLabel('', self)
        lay.addWidget(self._info)

    def _on_tool(self, btn):
        self.surface.tool = 'brush' if btn is self.btn_brush else 'eraser'

    def _on_radius(self, v):
        self.surface.radius = v

    def _on_reset(self):
        self.surface.undo_stack.append(self.surface.mask.copy())
        self.surface.mask = np.zeros_like(self.surface.mask, dtype=bool)
        self.surface.update()

    def _on_split(self):
        extra = self.surface.split()
        self.extra_pieces = extra
        self._info.setText(f'Split: {len(extra) + 1} parti (la principale resta in editor)')

    def _compute_results(self) -> List[Tuple[np.ndarray, List[int]]]:
        """[(mask_crop_locale, bbox_full)] — maschere e bbox coerenti."""
        results = []
        all_pieces = [self.surface.mask] + self.extra_pieces
        ox, oy = self.surface.ox, self.surface.oy
        for pc in all_pieces:
            pc = np.asarray(pc, dtype=bool)
            if not pc.any():
                continue
            x, y, w, h = bbox_of(pc)
            results.append((pc[y:y + h, x:x + w], [ox + x, oy + y, w, h]))
        return results

    def _on_ok(self):
        self.result_pieces = self._compute_results()
        self.accept()


def run_mask_editor(page_img, ins, parent=None):
    """Apre l'editor per l'istanza; ritorna (accepted, [ (mask, bbox) ])."""
    dlg = MaskEditorDialog(page_img, ins.mask, ins.bbox,
                           title=f'Edit mask #{ins.idx}', parent=parent)
    ok = dlg.exec() == QDialog.DialogCode.Accepted
    return ok, dlg.result_pieces
