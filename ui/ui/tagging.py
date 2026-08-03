# -*- coding: utf-8 -*-
"""Tagging semantico automatico delle parti applicate.

Euristica geometrica: individua busto (parte piu' grande centrale), testa
(sopra il busto), capelli (sopra/sovrapposta alla testa), arti (ai lati del
busto / sotto), e assegna left/right in base alla posizione del centro x
rispetto al centro del busto. Canonico: vedi TAG_SCHEMA.

Puro numpy: nessuna dipendenza Qt.
"""

from typing import Dict, List, Tuple

# tag canonici in ordine di disegno (dietro -> davanti), SEMPRE lowercase
TAG_SCHEMA: List[str] = [
    'hairb', 'hairf', 'head', 'face', 'eyel', 'eyer', 'browl', 'browr',
    'mouth', 'neck', 'torso', 'chest', 'waist', 'hip', 'skirt', 'tail',
    'arml', 'armr', 'handl', 'handr', 'legl', 'legr', 'footl', 'footr',
    'accessory', 'unknown',
]
_TAG_SET = set(TAG_SCHEMA)

BODY_TAGS = {'head', 'face', 'neck', 'torso', 'chest', 'waist', 'hip',
             'skirt', 'tail', 'arml', 'armr', 'handl', 'handr',
             'legl', 'legr', 'footl', 'footr'}


def is_canonical(tag: str) -> bool:
    return (tag or '').lower() in _TAG_SET


def auto_tag(parts: List[dict], img_shape: Tuple[int, int]) -> Dict[str, str]:
    """parts: [ {did, tag, x, y, w, h, area} ] (bbox full-image).

    Ritorna {did: tag_canonico}. I tag gia' canonici vengono mantenuti;
    gli altri vengono ristimati. Le parti con area < 0.5% dell'immagine e
    posizione interna restano 'unknown' se non stimabili.
    """
    H, W = img_shape
    total = float(H * W)
    out: Dict[str, str] = {}

    # parti con tag canonico gia' valido -> tenute ('unknown'/'none' no)
    for p in parts:
        t = (p.get('tag') or '').lower()
        if t in _TAG_SET and t not in ('unknown', 'none'):
            out[p['did']] = t

    def center(p):
        return (p['x'] + p['w'] / 2.0, p['y'] + p['h'] / 2.0)

    def area(p):
        return p.get('area', p['w'] * p['h'])

    # busto candidato: parte non taggata con area massima il cui centro e'
    # nella fascia verticale centrale e in orizzontale dentro il 60% centrale.
    # Esclude le maschere quasi-intera immagine (spesso background di SAM).
    untagged = [p for p in parts if p['did'] not in out]
    if not untagged:
        return out

    cands = [p for p in untagged if area(p) < 0.55 * total]
    if not cands:
        cands = untagged
    torso = None
    for p in sorted(cands, key=area, reverse=True):
        cx, cy = center(p)
        if 0.35 * H < cy < 0.85 * H and 0.2 * W < cx < 0.8 * W:
            torso = p
            break
    if torso is None and cands:
        torso = max(cands, key=area)
    if torso is not None:
        out[torso['did']] = 'torso'
    if torso is None:
        return out

    tx, ty = center(torso)
    tt = torso['y']
    tb = torso['y'] + torso['h']

    for p in untagged:
        if p['did'] == torso['did']:
            continue
        did, cx, cy = p['did'], center(p)[0], center(p)[1]
        ar = area(p)
        if ar < 0.003 * total:
            out.setdefault(did, 'unknown')
            continue
        above = cy < tt
        below = cy > tb
        beside = abs(cx - tx) > torso['w'] * 0.5
        if above:
            out[did] = 'hairf' if abs(cx - tx) <= torso['w'] * 0.8 else 'hairb'
        elif below:
            out[did] = 'legl' if cx < tx else 'legr'
        elif beside:
            out[did] = 'arml' if cx < tx else 'armr'
        else:
            out.setdefault(did, 'unknown')
    return out
