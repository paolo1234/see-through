# -*- coding: utf-8 -*-
"""Verifica che il percorso di on_candidate_activate non crashi:
Canvas.gv (QGraphicsView) deve esporre ensureVisible (era canvas.ensureVisible -> AttributeError).
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from qtpy.QtWidgets import QApplication
app = QApplication([])
from qtpy.QtCore import QRectF
from ui.canvas import Canvas

c = Canvas()
assert hasattr(c, 'gv'), 'Canvas.gv manca'
assert callable(getattr(c.gv, 'ensureVisible', None)), 'gv.ensureVisible manca'
assert hasattr(c, 'ensureVisible') is False or not callable(getattr(c, 'ensureVisible', None)), \
    'Canvas non deve avere ensureVisible diretto (il fix usa canvas.gv.ensureVisible)'
c.gv.ensureVisible(QRectF(0, 0, 50, 50))  # il call stesso, non deve sollevare
print('CANDIDATE ACTIVATE OK (canvas.gv.ensureVisible disponibile e chiamabile)')
