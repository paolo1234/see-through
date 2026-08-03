"""Fase 5: operazioni pure sulle maschere di istanza (numpy, testabili).

Convenzione: Instance.mask e' crop-locale (bool, shape = bbox h x w),
Instance.bbox = [x, y, w, h] in coordinate immagine full-size.
"""
from typing import List, Tuple

import cv2
import numpy as np


def full_mask(ins, H: int, W: int) -> np.ndarray:
    """Ricostruisce la maschera full-size dell'istanza."""
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
    return m


def bbox_of(mask: np.ndarray) -> List[int]:
    """[x, y, w, h] della maschera (crop-locale o full-size)."""
    m = np.asarray(mask, dtype=np.uint8)
    if not m.any():
        return [0, 0, 0, 0]
    ys, xs = np.nonzero(m)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def paint(mask: np.ndarray, points, labels, radius: int = 8) -> np.ndarray:
    """Dipinge cerchi sulla maschera crop-locale.
    points: list[(x, y)] locali; labels: list[int] (1=foreground, 0=erase).
    Restituisce una NUOVA maschera (l'originale non viene toccata)."""
    out = np.asarray(mask).copy()
    r = max(2, int(radius))
    h, w = out.shape[:2]
    for (x, y), lab in zip(points, labels):
        cx, cy = int(x), int(y)
        # bounding box del cerchio, clippato
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        if lab == 1:
            out[y0:y1, x0:x1] |= disk
        else:
            out[y0:y1, x0:x1] &= ~disk
    return out


def split_components(mask: np.ndarray, min_area: int = 8) -> List[np.ndarray]:
    """Divide una maschera crop-locale in componenti connesse separate.
    Restituisce maschere crop-locale (ciascuna col proprio bbox)."""
    m = np.asarray(mask, dtype=np.uint8)
    if not m.any():
        return []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = (int(v) for v in stats[i][:4])
        piece = (labels[y:y + h, x:x + w] == i)
        comps.append(piece)
    comps.sort(key=lambda c: -int(c.sum()))
    return comps


def merge_masks(masks: List[np.ndarray], bboxes: List[List[int]], H: int, W: int) -> Tuple[np.ndarray, List[int]]:
    """Unisce piu' istanze (maschere crop-locali) in una sola full-mask + bbox."""
    full = np.zeros((H, W), dtype=bool)
    for mc, bb in zip(masks, bboxes):
        x, y, w, h = (int(v) for v in bb)
        mc = np.asarray(mc, dtype=bool)
        if mc.shape != (h, w):
            mc = cv2.resize(mc.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST) > 0
        x2 = min(x + w, W)
        y2 = min(y + h, H)
        if y2 > y and x2 > x:
            full[y:y2, x:x2] |= mc[: y2 - y, : x2 - x]
    if not full.any():
        return np.zeros((0, 0), dtype=bool), [0, 0, 0, 0]
    x, y, w, h = bbox_of(full)
    return full[y:y + h, x:x + w], [x, y, w, h]


def morphology(mask: np.ndarray, op: str, ksize: int = 3) -> np.ndarray:
    """erode | dilate | open | close sulla maschera crop-locale."""
    m = np.asarray(mask, dtype=np.uint8)
    k = max(2, int(ksize))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fn = {'erode': cv2.MORPH_ERODE, 'dilate': cv2.MORPH_DILATE,
          'open': cv2.MORPH_OPEN, 'close': cv2.MORPH_CLOSE}.get(op)
    if fn is None:
        raise ValueError(f'op sconosciuta: {op}')
    return cv2.morphologyEx(m, fn, kernel,
                            borderType=cv2.BORDER_CONSTANT,
                            borderValue=0).astype(bool)
