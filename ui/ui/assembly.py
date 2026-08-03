"""Fase 3: assembly — da istanze candidato a Drawable taggati.

Strategia: per ogni istanza (maschera full-size) si vota il tag della parte
con la maggior IoU contro i drawable gia' presenti nel modello della pagina
(il paper usa un voting analogo per aggregare regioni in parti). Le istanze
senza overlap sufficiente restano 'unknown' e sono correggibili a mano.
"""
from typing import List, Tuple

import cv2
import numpy as np

from live2d.scrap_model import Drawable

MIN_IOU = 0.15


def full_mask_of_drawable(d: Drawable, H: int, W: int):
    """Maschera booleana full-size del drawable, allineata con l'immagine."""
    vm = d.visible_mask
    if vm is None or not isinstance(vm, np.ndarray) or vm.dtype == object:
        return None
    if not vm.any():
        return None
    fm = np.zeros((H, W), dtype=bool)
    x, y = int(d.x), int(d.y)
    mh, mw = vm.shape[:2]
    y2 = min(y + mh, H)
    x2 = min(x + mw, W)
    if y2 > y and x2 > x:
        fm[y:y2, x:x2] = vm[: y2 - y, : x2 - x]
    return fm


def vote_tags(lmodel, instances: List, img: np.ndarray, min_iou: float = MIN_IOU) -> List[Tuple]:
    """[(instance, tag, iou, best_did)] — tag = parte con max IoU (>= min_iou)."""
    H, W = img.shape[:2]
    pool = []
    for d in lmodel.valid_drawables():
        if getattr(d, 'tag', None) is None:
            continue
        fm = full_mask_of_drawable(d, H, W)
        if fm is not None:
            pool.append((d, fm))

    results: List[Tuple] = []
    for ins in instances:
        # Instance.mask e' crop-locale (bbox-relative): ricostruisci full-size
        m = np.zeros((H, W), dtype=bool)
        x, y, w, h = (int(v) for v in ins.bbox)
        mc = np.asarray(ins.mask, dtype=bool)
        if mc.shape != (h, w):
            mc = cv2.resize(mc.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST) > 0
        x2 = min(x + w, W)
        y2 = min(y + h, H)
        if y2 > y and x2 > x:
            m[y:y2, x:x2] = mc[: y2 - y, : x2 - x]
        best_tag, best_iou, best_did = 'unknown', 0.0, None
        for d, fm in pool:
            inter = int(np.logical_and(m, fm).sum())
            union = int(np.logical_or(m, fm).sum())
            if union <= 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou, best_tag, best_did = iou, d.tag, d.did
        results.append((ins, best_tag if best_iou >= min_iou else 'unknown',
                        best_iou, best_did))
    return results


def make_drawable(ins, tag: str, img: np.ndarray, page: str) -> Drawable:
    """Costruisce un Drawable editabile a partire da un'istanza candidata."""
    cutout = ins.get_cutout(img)
    if cutout is None:
        return None
    H, W = img.shape[:2]
    x, y, w, h = (int(v) for v in ins.bbox)
    d = Drawable(img=cutout, draw_order=0, src_path=None,
                 crop_xyxy=[x, y, w, h], pad_drawable_img=False,
                 seg_type='body_part_tag', src_size=(H, W))
    d.x, d.y = x, y
    d.set_tag(tag)
    d.did = f'inst://{page}/{ins.idx}'
    d.idx = ins.idx

    # Instance.mask e' crop-locale: deve combaciare col bbox
    mc = np.asarray(ins.mask, dtype=bool)
    if mc.shape != (h, w):
        mc = cv2.resize(mc.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST) > 0
    if mc.size == 0:
        return None
    d.visible_mask = mc
    d.area = int(mc.sum())
    d.final_visible_mask = mc
    d.final_visible_area = d.area
    return d
