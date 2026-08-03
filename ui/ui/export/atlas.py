# -*- coding: utf-8 -*-
"""Bake di cycle animation in sprite atlas a righe (una riga per stato) +
manifest.json con frame_layout, compatibile con il formato sprite-gen:

    {sheetWidth, sheetHeight, cellWidth, cellHeight,
     rows: {<state>: [ {x,y,w,h}, ... ]},          # frame_layout (ordine di playback)
     animation: {cellWidth, cellHeight, columns,
                 rows: {<state>: {row, frames, fps, durations_ms, loop}}}}

Un frame per cella; ogni cella ha la stessa dimensione (motion box + padding)
cosi' le righe sono perfettamente allineate e i motori campionano i rect.

Puro numpy/PIL: nessuna dipendenza Qt.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from ..anim.preview import render_pose, bbox_alpha
from ..anim.pose import Pose


def _motion_box(frames: List[np.ndarray]) -> Optional[tuple]:
    """bbox unione dei pixel non trasparenti di tutti i frame."""
    u = None
    for f in frames:
        b = bbox_alpha(f)
        if b is None:
            continue
        if u is None:
            u = list(b)
        else:
            u[0] = min(u[0], b[0]); u[1] = min(u[1], b[1])
            u[2] = max(u[2], b[2]); u[3] = max(u[3], b[3])
    return tuple(u) if u else None


def bake_atlas(parts: List[dict], cycles: Dict[str, List[Pose]],
               size: tuple, padding: int = 8,
               fps: Optional[Dict[str, int]] = None,
               base: Optional[np.ndarray] = None):
    """cycles: {state: [Pose, ...]}; size = (W, H) canvas.

    Ritorna (strip RGBA np.ndarray, manifest dict, report dict).
    """
    rendered: Dict[str, List[np.ndarray]] = {}
    for state, poses in cycles.items():
        rendered[state] = [render_pose(parts, p, size, base=base) for p in poses]

    mb = None
    for state, frames in rendered.items():
        b = _motion_box(frames)
        if mb is None:
            mb = b
        elif b is not None:
            mb = (min(mb[0], b[0]), min(mb[1], b[1]),
                  max(mb[2], b[2]), max(mb[3], b[3]))
    if mb is None:
        raise ValueError('Nessun pixel visibile nei frame: applica prima le parti.')

    cell_w = mb[2] - mb[0] + 2 * padding
    cell_h = mb[3] - mb[1] + 2 * padding

    states = list(cycles.keys())
    ncols = max(len(rendered[s]) for s in states)
    sheet_w = ncols * cell_w
    sheet_h = len(states) * cell_h
    strip = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)

    layout_rows: Dict[str, list] = {}
    anim_rows: Dict[str, dict] = {}
    report_cells = []
    for row_idx, state in enumerate(states):
        frames = rendered[state]
        rects = []
        for col, frame in enumerate(frames):
            bx = bbox_alpha(frame)
            # centra il contenuto del frame nella cella
            ox = (cell_w - (bx[2] - bx[0])) // 2 - bx[0]
            oy = (cell_h - (bx[3] - bx[1])) // 2 - bx[1]
            x0, y0 = col * cell_w + max(0, ox), row_idx * cell_h + max(0, oy)
            x1, y1 = x0 + (bx[2] - bx[0]), y0 + (bx[3] - bx[1])
            x1 = min(x1, sheet_w); y1 = min(y1, sheet_h)
            if x1 > x0 and y1 > y0:
                strip[y0:y1, x0:x1] = frame[bx[1]:bx[3], bx[0]:bx[2]][:y1 - y0, :x1 - x0]
            rect = {'x': col * cell_w, 'y': row_idx * cell_h,
                    'w': cell_w, 'h': cell_h}
            rects.append(rect)
            report_cells.append({'state': state, 'frame': col, **rect})
        layout_rows[state] = rects
        st_fps = int((fps or {}).get(state, 8) or 8)
        anim_rows[state] = {
            'row': row_idx,
            'frames': len(frames),
            'fps': st_fps,
            'durations_ms': [max(1, round(1000.0 / st_fps))] * len(frames),
            'loop': True,
        }

    manifest = {
        'engine': 'see-through-cycle',
        'sheetWidth': sheet_w,
        'sheetHeight': sheet_h,
        'cellWidth': cell_w,
        'cellHeight': cell_h,
        'rows': layout_rows,
        'animation': {
            'cellWidth': cell_w,
            'cellHeight': cell_h,
            'columns': ncols,
            'rows': anim_rows,
        },
    }
    report = {
        'ok': True,
        'states': states,
        'motion_box': list(mb),
        'cell': [cell_w, cell_h],
        'sheet': [sheet_w, sheet_h],
        'cells': report_cells,
        'padding': padding,
    }
    return strip, manifest, report


def save_atlas(strip: np.ndarray, manifest: dict, out_dir: str,
               sheet_name: str = 'sprite-sheet-alpha.png') -> str:
    """Salva strip + manifest nel formato sprite-gen e ritorna il percorso sheet."""
    os.makedirs(out_dir, exist_ok=True)
    sheet_path = os.path.join(out_dir, sheet_name)
    Image.fromarray(strip, 'RGBA').save(sheet_path)
    manifest['atlas'] = sheet_path
    manifest['manifest'] = os.path.join(out_dir, 'manifest.json')
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return sheet_path
