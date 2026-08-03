# -*- coding: utf-8 -*-
"""Pose di una parte e helpers per i pivot (joint anchor).

Convenzioni:
  - coordinate immagine: y verso il BASSO (come canvas/immagine),
  - rotazione in gradi, POSITIVA in senso orario (rotazione vista su canvas),
  - pivot in coordinate FULL-IMAGE: e' il punto attorno a cui la parte
    ruota/scala e che riceve la traslazione.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PartPose:
    """Trasformazione di una singola parte a un dato istante.

    La composizione e' (vedi preview.render_pose):
        q = R(angle) * S(scale) * (p - pivot) + pivot + (dx, dy)
    dove R e' la rotazione oraria in gradi e S la scala attorno al pivot.
    """
    pivot: tuple = (0.0, 0.0)
    angle: float = 0.0       # gradi, orario (canvas y-down)
    scale_x: float = 1.0
    scale_y: float = 1.0
    dx: float = 0.0          # traslazione in px (full-image), dopo rotazione
    dy: float = 0.0
    visible: bool = True
    opacity: float = 1.0     # 0..1
    z: int = 0               # ordine di disegno (dalla lista parti, non usato qui)

    def copy(self) -> 'PartPose':
        return PartPose(tuple(self.pivot), self.angle, self.scale_x, self.scale_y,
                        self.dx, self.dy, self.visible, self.opacity, self.z)


class Pose(Dict[str, PartPose]):
    """did -> PartPose per un istante."""

    def identity_for(self, did: str, x: float, y: float, w: float, h: float,
                     pivot: tuple = None) -> 'Pose':
        self[did] = PartPose(pivot=pivot or pivot_for_tag('', x, y, w, h))
        return self


# ---------------------------------------------------------------------------
# Pivot per tag (euristiche di giunzione anatomica). Canvas y-down.
# ---------------------------------------------------------------------------
LIMB_TAGS = {'arml', 'armr', 'legl', 'legr', 'handl', 'handr', 'footl', 'footr',
             'hairf', 'hairb', 'tail', 'skirt'}
NECK_TAGS = {'head', 'face', 'neck'}
TORSOTAGS = {'torso', 'chest', 'waist', 'hip'}
FACE_TAGS = {'eyel', 'eyer', 'browl', 'browr', 'mouth'}


def pivot_for_tag(tag: str, x: float, y: float, w: float, h: float) -> tuple:
    """Joint anchor della parte in base al tag (bbox in coordinate full-image)."""
    t = (tag or 'unknown').lower()
    if t in LIMB_TAGS:
        return (x + w / 2.0, y)              # giunto in ALTO (spalla/anca/collo)
    if t in NECK_TAGS:
        return (x + w / 2.0, y + h)          # collo in BASSO
    if t in TORSOTAGS:
        return (x + w / 2.0, y + h)          # bacino in BASSO
    if t in FACE_TAGS:
        return (x + w / 2.0, y + h / 2.0)    # centro (micro-espressioni)
    return (x + w / 2.0, y + h / 2.0)
