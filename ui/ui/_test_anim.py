# -*- coding: utf-8 -*-
"""Test dei moduli puri di animazione: tagging, cycles, preview, atlas.
Run: PYTHONPATH=".;common;annotators" .venv-st/Scripts/python.exe ui/ui/_test_anim.py
"""
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ui.anim.pose import Pose, PartPose, pivot_for_tag
from ui.anim.cycles import CycleParams, generate_cycle
from ui.anim.preview import render_pose, transform_part
from ui.export.atlas import bake_atlas, save_atlas
from ui.tagging import auto_tag, is_canonical


def make_part(did, tag, x, y, w, h, color=(200, 60, 60, 255)):
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 0] = color[0]
    img[..., 1] = color[1]
    img[..., 2] = color[2]
    img[..., 3] = color[3]
    return {'did': did, 'tag': tag, 'x': x, 'y': y, 'w': w, 'h': h,
            'area': w * h, 'img': img}


# ---------- 1) tagging ----------
def test_tagging():
    W, H = 512, 512
    parts = [
        make_part('t', None, 206, 190, 100, 120, (0, 0, 255, 255)),   # torso
        make_part('h', None, 218, 90, 76, 90, (255, 0, 0, 255)),      # head
        make_part('al', None, 120, 200, 40, 90, (0, 255, 0, 255)),    # armL
        make_part('ar', None, 352, 200, 40, 90, (0, 255, 0, 255)),    # armR
        make_part('ll', None, 216, 320, 36, 110, (255, 255, 0, 255)), # legL
        make_part('lr', None, 260, 320, 36, 110, (255, 255, 0, 255)), # legR
    ]
    tags = auto_tag(parts, (H, W))
    assert tags['t'] == 'torso', tags
    assert tags["h"] == "hairf", tags          # sopra il torso, vicino al centro
    assert tags['al'] == 'arml', tags
    assert tags['ar'] == 'armr', tags
    assert tags['ll'] == 'legl', tags
    assert tags['lr'] == 'legr', tags
    # tag canonico gia' presente viene mantenuto
    parts[1]['tag'] = 'head'
    tags = auto_tag(parts, (H, W))
    assert tags['h'] == 'head', tags
    assert all(is_canonical(v) for v in tags.values()), tags
    print('tagging OK', tags)


# ---------- 2) pivot ----------
def test_pivots():
    assert pivot_for_tag('armL', 10, 20, 40, 80) == (30.0, 20.0)      # top-center
    assert pivot_for_tag('head', 10, 20, 40, 80) == (30.0, 100.0)     # bottom-center
    assert pivot_for_tag('torso', 10, 20, 40, 80) == (30.0, 100.0)    # bottom-center
    print('pivots OK')


# ---------- 3) cycles: seamless + movimento ----------
def test_cycles():
    parts = [
        make_part('t', 'torso', 206, 190, 100, 120),
        make_part('h', 'head', 218, 90, 76, 90),
        make_part('al', 'armL', 120, 200, 40, 90),
        make_part('ar', 'armR', 352, 200, 40, 90),
        make_part('ll', 'legL', 216, 320, 36, 110),
        make_part('lr', 'legR', 260, 320, 36, 110),
    ]
    # walk
    p = CycleParams.defaults('walk')
    frames = generate_cycle(parts, p)
    assert len(frames) == p.frames
    # seamless: campiona t=0 e t=T -> stessa pose
    f0 = frames[0]
    T = p.duration_s
    from ui.anim.cycles import _phase
    ph = _phase(T, T)
    assert abs(ph - 2 * np.pi) < 1e-9
    # frame N-1 = t=(N-1)/N*T; il "prossimo" sarebbe t=T = identico a t=0.
    # Verifichiamo periodicita': pose(t=0) == pose(t=T) via sampling diretto
    def pose_at(t):
        from ui.anim.cycles import _phase
        from math import sin
        phi = _phase(t, T)
        out = {}
        for p_ in parts:
            tag = p_['tag']
            pp = PartPose(pivot=pivot_for_tag(tag, p_['x'], p_['y'], p_['w'], p_['h']))
            if tag in ('legL',):
                pp.angle = p.leg_swing * sin(phi + 3.141592653589793)
            elif tag in ('legR',):
                pp.angle = p.leg_swing * sin(phi)
            out[p_['did']] = pp
        return out
    a, b = pose_at(0.0), pose_at(T)
    for k in a:
        assert abs(a[k].angle - b[k].angle) < 1e-6, k
    # le gambe si muovono davvero (ampiezza > 0 su qualche frame)
    swings = {k: max(abs(f[k].angle) for f in frames) for k in ('ll', 'lr', 'al', 'ar')}
    assert swings['ll'] > 10 and swings['lr'] > 10, swings
    assert swings['al'] > 10 and swings['ar'] > 10, swings
    # braccio anti-fase rispetto alla gamba omonima (stesso seno, segno opposto,
    # ampiezze diverse: leg_swing vs arm_swing)
    f_mid = frames[1]
    assert f_mid['ll'].angle * f_mid['al'].angle < 0, (f_mid['ll'].angle, f_mid['al'].angle)
    ratio = abs(f_mid['ll'].angle / f_mid['al'].angle)
    assert abs(ratio - p.leg_swing / p.arm_swing) < 1e-6, ratio
    # bob a doppia frequenza sul torso
    bob_y = [f['t'].dy for f in frames]
    assert max(abs(v) for v in bob_y) > 1.0, bob_y
    print('cycles OK (swings %s, bob %.1f)' % (swings, max(bob_y)))


# ---------- 4) preview: identita' e rotazione attorno al pivot ----------
def test_preview():
    W, H = 512, 512
    parts = [make_part('ll', 'legL', 200, 300, 40, 100)]
    parts[0]['img'][20:80, 10:30] = (0, 0, 0, 255)  # marca una zona
    pose = Pose()
    pivot = pivot_for_tag('legL', 200, 300, 40, 100)   # (220, 300)
    pose['ll'] = PartPose(pivot=pivot)
    img0 = render_pose(parts, pose, (W, H))
    # in identita' la parte deve stare al suo bbox
    assert (img0[310, 220, 3] > 0) and (img0[310, 220, :3] != 0).any()  # dentro il leg
    assert img0[299, 220, 3] == 0  # sopra il leg (y<300) trasparente
    # rotazione di 90°: il contenuto ruota attorno al pivot
    pose['ll'] = PartPose(pivot=pivot, angle=90.0)
    img90 = render_pose(parts, pose, (W, H))
    # dopo 90° oraria la striscia verticale (200,300,40,100) diventa
    # orizzontale: bbox atteso ~x 120-220, y 280-320 -> centro (170, 300)
    assert img90[300, 170, 3] > 0, img90[290:310, 160:180, 3].max()
    print('preview OK')


# ---------- 5) atlas: struttura + manifest sprite-gen ----------
def test_atlas():
    W, H = 512, 512
    parts = [
        make_part('t', 'torso', 206, 190, 100, 120),
        make_part('h', 'head', 218, 90, 76, 90),
        make_part('al', 'armL', 120, 200, 40, 90),
        make_part('ar', 'armR', 352, 200, 40, 90),
        make_part('ll', 'legL', 216, 320, 36, 110),
        make_part('lr', 'legR', 260, 320, 36, 110),
    ]
    cycles = {}
    for kind in ('idle', 'walk', 'run'):
        p = CycleParams.defaults(kind)
        cycles[kind] = generate_cycle(parts, p)
    strip, manifest, report = bake_atlas(parts, cycles, (W, H), padding=8)
    assert manifest['rows'].keys() == {'idle', 'walk', 'run'}, manifest['rows'].keys()
    ncols = max(len(v) for v in cycles.values())
    assert strip.shape[0] == 3 * manifest['cellHeight']
    assert strip.shape[1] == ncols * manifest['cellWidth']
    assert manifest['cellWidth'] > 0 and manifest['cellHeight'] > 0
    for st, rects in manifest['rows'].items():
        assert len(rects) == len(cycles[st])
        for r in rects:
            assert (r['x'], r['y'], r['w'], r['h']) == \
                (r['x'], r['y'], manifest['cellWidth'], manifest['cellHeight'])
            cell = strip[r['y']:r['y'] + r['h'], r['x']:r['x'] + r['w']]
            assert (cell[..., 3] > 8).any(), (st, r)  # ogni cella non e' vuota
    # salvataggio
    d = tempfile.mkdtemp(prefix='st_anim_')
    sheet = save_atlas(strip, manifest, d)
    assert os.path.isfile(sheet) and os.path.isfile(os.path.join(d, 'manifest.json'))
    import json
    m2 = json.load(open(os.path.join(d, 'manifest.json'), encoding='utf-8'))
    assert m2['animation']['rows']['walk']['fps'] == 8
    assert m2['animation']['rows']['walk']['loop'] is True
    print('atlas OK sheet=%s cell=%dx%d states=%s' % (
        os.path.basename(sheet), manifest['cellWidth'], manifest['cellHeight'],
        list(manifest['rows'].keys())))


if __name__ == '__main__':
    test_tagging()
    test_pivots()
    test_cycles()
    test_preview()
    test_atlas()
    print('TEST ANIM OK')
