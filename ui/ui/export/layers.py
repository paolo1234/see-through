# -*- coding: utf-8 -*-
"""Export dei layer della pagina corrente: singole parti (PNG con alpha)
oppure un file Photoshop PSD con un layer per parte + sfondo.

Dipendenze opzionali: psd-tools (solo per l'export PSD). Il percorso PNG
e' usabile da qualsiasi engine (Godot, Unity, Spine...) come base per
rigging/spine: ogni parte e' un'immagine ritagliata sul suo bbox con alpha.
"""

import os

import numpy as np

from ..logger import logger as LOGGER


def collect_parts(proj, drop_background: bool = True):
    """Raccoglie le parti applicate della pagina corrente.

    Stessa logica di anim_dialog._load_parts ma SENZA sovrascrivere i tag
    (usa il tag canonico della drawable; 'unknown' se assente).
    Ritorna list[dict] did, tag, x, y, w, h, area, img (RGBA).
    """
    if not getattr(proj, 'model_valid', False) or proj.current_image is None:
        return []
    H, W = proj.current_image.shape[:2]
    parts = []
    for dr in proj.l2dmodel.valid_drawables():
        try:
            crop = np.asarray(dr.get_img(), dtype=np.uint8)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f'layers: skip {dr.did}: {e}')
            continue
        if crop.ndim == 2:
            crop = np.stack([crop] * 3 + [np.full_like(crop, 255)], axis=-1)
        elif crop.shape[2] == 3:
            crop = np.concatenate(
                [crop, np.full(crop.shape[:2] + (1,), 255, np.uint8)], axis=-1)
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            continue
        area = int(getattr(dr, 'area', w * h) or 0)
        if drop_background and area > 0.55 * H * W:
            continue  # maschera quasi-intera immagine = background di SAM
        parts.append({'did': dr.did, 'tag': (dr.tag or '').lower() or 'unknown',
                      'x': int(getattr(dr, 'x', 0)), 'y': int(getattr(dr, 'y', 0)),
                      'w': w, 'h': h, 'area': area, 'img': crop})
    return parts


def _sanitize(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ('-', '_', '.'):
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out)


def _write_png(path: str, img_rgba: np.ndarray):
    from PIL import Image
    Image.fromarray(img_rgba, 'RGBA').save(path)


def export_layers_png(proj, parts: list, out_dir: str):
    """Scrive background.png, preview.png e NN_<tag>_<did>.png per parte.

    Ritorna la lista dei file scritti.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = proj.current_image  # RGB
    H, W = base.shape[:2]
    bg = np.concatenate([base, np.full(base.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    files = []
    bg_path = os.path.join(out_dir, 'background.png')
    _write_png(bg_path, bg)
    files.append(bg_path)

    # composita (parti sopra il fondo, in ordine di drawable)
    comp = bg.copy()
    for p in parts:
        x0, y0, w, h = p['x'], p['y'], p['w'], p['h']
        img = p['img']
        a = img[..., 3:4].astype(np.float64) / 255.0
        region = comp[y0:y0 + h, x0:x0 + w]
        region[..., :4] = (img[..., :4].astype(np.float64) * a
                           + region[..., :4].astype(np.float64) * (1 - a)).astype(np.uint8)

    preview_path = os.path.join(out_dir, 'preview.png')
    _write_png(preview_path, comp)
    files.append(preview_path)

    for i, p in enumerate(parts):
        did = _sanitize(p['did'].replace('inst://', '').replace('\\\\', '_'))
        name = f'{i:02d}_{p["tag"]}_{did}.png'
        path = os.path.join(out_dir, name)
        _write_png(path, p['img'])
        files.append(path)
    LOGGER.info(f'Layers PNG: {len(files)} file in {out_dir}')
    return files


def export_layers_psd(proj, parts: list, path: str) -> int:
    """Scrive un PSD RGBA con layer background + uno per parte.

    Ritorna il numero di layer scritti (background escluso).
    """
    from psd_tools import PSDImage
    from PIL import Image as PILImage
    base = proj.current_image
    H, W = base.shape[:2]
    bg = np.concatenate([base, np.full(base.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    psd = PSDImage.new('RGBA', (W, H))
    psd.create_pixel_layer(image=PILImage.fromarray(bg, 'RGBA'), name='background')
    n = 0
    for i, p in enumerate(parts):
        did = _sanitize(p['did'].replace('inst://', '').replace('\\\\', '_'))
        name = f'{i:02d}_{p["tag"]}_{did}'
        try:
            psd.create_pixel_layer(image=PILImage.fromarray(p['img'], 'RGBA'),
                                   name=name, top=int(p['y']), left=int(p['x']))
            n += 1
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f'psd layer {name}: {e}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    psd.save(path)
    LOGGER.info(f'PSD: {path} ({n + 1} layer)')
    return n
