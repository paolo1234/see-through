# -*- coding: utf-8 -*-
"""Export PNG sequence per stato (idle/walk/run) dai frame generati.

Ogni frame è l'immagine full-page RGBA renderizzata con la pose. Le
sequenze sono il formato più interoperabile (import in qualsiasi engine,
editor video o sprite-gen).
"""

import os

import numpy as np

from ..logger import logger as LOGGER


def export_sequences(parts: list, cycles: dict, size: tuple, out_dir: str,
                     prefix: str = 'frame') -> list:
    """cycles: dict kind -> list[Pose]. Scrive <out>/<kind>/<prefix>_NNN.png.

    Ritorna la lista dei file scritti.
    """
    from ..anim.preview import render_pose
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for kind, frames in (cycles or {}).items():
        d = os.path.join(out_dir, kind)
        os.makedirs(d, exist_ok=True)
        for i, pose in enumerate(frames):
            img = render_pose(parts, pose, size)
            if img.ndim == 3 and img.shape[2] == 3:
                img = np.concatenate(
                    [img, np.full(img.shape[:2] + (1,), 255, np.uint8)], axis=-1)
            path = os.path.join(d, f'{prefix}_{i:03d}.png')
            Image.fromarray(img, 'RGBA').save(path)
            files.append(path)
    LOGGER.info(f'PNG sequences: {len(files)} file in {out_dir}')
    return files
