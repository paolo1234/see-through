# -*- coding: utf-8 -*-
"""Render di una Pose: composita le parti (RGBA) su un canvas trasparente
con trasformazione affine attorno al pivot (rotazione oraria, scala, trasl.).

Puro numpy/cv2 -> riusabile da preview UI, bake atlas e Colab.
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

from .pose import PartPose, Pose


def transform_part(crop: np.ndarray, pivot: tuple, angle_deg: float,
                   sx: float = 1.0, sy: float = 1.0,
                   dx: float = 0.0, dy: float = 0.0,
                   off: tuple = (0, 0)):
    """Warp del crop attorno al pivot (coordinate globali).

    off = (x, y) posizione del crop nell'immagine piena: la trasformazione
    agisce su coordinate full-image (crop pixel (i,j) == full (x+i, y+j)).
    Ritorna (warped RGBA, gx0, gy0): warped va composto sul canvas a (gx0, gy0).
    """
    h, w = crop.shape[:2]
    ox, oy = off
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    S = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)
    p0 = np.asarray(pivot, dtype=np.float64)
    d = np.asarray([dx, dy], dtype=np.float64)
    RS = R @ S
    # bbox OUTPUT degli angoli del crop in coordinate FULL-IMAGE
    corners = np.array([[ox, oy], [ox + w, oy], [ox, oy + h], [ox + w, oy + h]],
                       dtype=np.float64)
    fwd = (RS @ (corners - p0).T).T + p0 + d
    gx0, gy0 = np.floor(fwd.min(axis=0)).astype(int)
    gx1, gy1 = np.ceil(fwd.max(axis=0)).astype(int)
    dw, dh = max(1, gx1 - gx0), max(1, gy1 - gy0)

    if crop.dtype != np.uint8 or crop.shape[2] != 4:
        crop = np.asarray(crop, dtype=np.uint8)
        if crop.ndim == 2:
            crop = np.stack([crop] * 3 + [np.full_like(crop, 255)], axis=-1)
        elif crop.shape[2] == 3:
            crop = np.concatenate([crop, np.full(crop.shape[:2] + (1,), 255, np.uint8)], axis=-1)

    # matrice inversa su coordinate GLOBALI -> il buffer dst e' posizionato a (gx0, gy0)
    RSinv = np.linalg.inv(RS)
    # input_crop = RSinv @ (q - p0 - d) + p0 - off
    b = -RSinv @ (p0 + d) + p0 - np.asarray([ox, oy], dtype=np.float64)
    A = np.hstack([RSinv, b.reshape(2, 1)])
    # shift: dst(i,j) == globale(gx0+i, gy0+j) -> compongo la traslazione
    A[0, 2] += RSinv[0, 0] * gx0 + RSinv[0, 1] * gy0
    A[1, 2] += RSinv[1, 0] * gx0 + RSinv[1, 1] * gy0

    warped = cv2.warpAffine(crop, A, (dw, dh),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT)
    return warped, gx0, gy0


def render_pose(parts: List[dict], pose: Optional[Pose],
                size: tuple, base: Optional[np.ndarray] = None) -> np.ndarray:
    """Composita le parti (dict: did, tag, x, y, img RGBA) secondo la pose.

    size = (W, H) canvas; base opzionale RGBA disegnata sotto (es. shadow).
    Ritorna RGBA uint8 (W, H).
    """
    W, H = size
    if base is not None and base.shape[:2] == (H, W):
        canvas = np.asarray(base, dtype=np.uint8).copy()
    else:
        canvas = np.zeros((H, W, 4), dtype=np.uint8)
    pose = pose or Pose()
    for part in parts:
        pp: PartPose = pose.get(part['did'])
        if pp is None:
            continue
        if not pp.visible or pp.opacity <= 0.0:
            continue
        crop = np.asarray(part['img'], dtype=np.uint8)
        if crop.ndim == 2:
            crop = np.stack([crop] * 3 + [np.full_like(crop, 255)], axis=-1)
        elif crop.shape[2] == 3:
            crop = np.concatenate([crop, np.full(crop.shape[:2] + (1,), 255, np.uint8)], axis=-1)
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        warped, gx0, gy0 = transform_part(
            crop, pp.pivot, pp.angle, pp.scale_x, pp.scale_y, pp.dx, pp.dy,
            off=(part.get('x', 0), part.get('y', 0)))
        if pp.opacity < 1.0 and pp.opacity > 0.0:
            warped = warped.copy()
            warped[..., 3] = (warped[..., 3].astype(np.float64) * pp.opacity).astype(np.uint8)
        _alpha_composite(canvas, warped, gx0, gy0)
    return canvas


def _alpha_composite(canvas: np.ndarray, warped: np.ndarray, gx0: int, gy0: int):
    H, W = canvas.shape[:2]
    x0, y0 = max(0, gx0), max(0, gy0)
    x1 = min(W, gx0 + warped.shape[1])
    y1 = min(H, gy0 + warped.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    src = warped[y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0]
    dst = canvas[y0:y1, x0:x1]
    a = (src[..., 3:4].astype(np.float64) / 255.0)
    canvas[y0:y1, x0:x1] = (src[..., :4].astype(np.float64) * a
                            + dst[..., :4].astype(np.float64) * (1.0 - a)).astype(np.uint8)


def bbox_alpha(img: np.ndarray) -> tuple:
    """bbox (x0, y0, x1, y1) dei pixel non trasparenti; None se vuota."""
    alpha = img[..., 3] > 8
    if not alpha.any():
        return None
    ys, xs = np.where(alpha)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
