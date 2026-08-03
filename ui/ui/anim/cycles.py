# -*- coding: utf-8 -*-
"""Cycle animation parametriche (idle / walk / run).

Genera N frame di una Pose per parte, campionando funzioni periodiche di
periodo T -> il loop primo/ultimo frame e' SEAMLESS per costruzione.

Filosofia: niente keyframe manuali per i cicli base. Ogni parte con tag
riconosciuto riceve una legge del moto parametrica (seno con fase), con
ampiezze esposte. Le parti non taggate restano identita' (statiche).

Convenzioni canvas: y-down, rotazione oraria positiva. Se il personaggio
guarda verso +x, una rotazione positiva di una gamba/braccio spinge il piede/
mano verso AVANTI (+x).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .pose import PartPose, Pose, pivot_for_tag, TORSOTAGS

# tag -> gruppo funzionale
LEG_TAGS = {'legl', 'legr'}
ARM_TAGS = {'arml', 'armr', 'handl', 'handr'}
HEAD_TAGS = {'head', 'face', 'neck'}


@dataclass
class CycleParams:
    """Parametri del generatore. Tutte le ampiezze in gradi/px."""

    kind: str = 'walk'            # idle | walk | run
    duration_s: float = 1.0       # durata del ciclo
    frames: int = 8               # frame per ciclo (loop: N-1 -> 0)

    # ampiezze
    leg_swing: float = 28.0       # rotazione gamba (deg)
    arm_swing: float = 24.0       # rotazione braccio (deg)
    bob: float = 6.0              # rimbalzo verticale del corpo (px)
    lean: float = 4.0             # inclinazione busto in avanti (deg)
    breath: float = 0.012         # respiro: variazione scalaY torso (0..1)
    head_nod: float = 0.0         # micro cenno testa (deg)

    @classmethod
    def defaults(cls, kind: str) -> 'CycleParams':
        p = cls(kind=kind)
        if kind == 'idle':
            p.duration_s, p.frames = 4.0, 8
            p.leg_swing = p.arm_swing = 2.0
            p.bob = 1.0
            p.lean = 0.0
            p.head_nod = 1.0
        elif kind == 'walk':
            p.duration_s, p.frames = 1.0, 8
            p.leg_swing, p.arm_swing = 30.0, 26.0
            p.bob = 6.0
            p.lean = 5.0
        elif kind == 'run':
            p.duration_s, p.frames = 0.7, 8
            p.leg_swing, p.arm_swing = 46.0, 40.0
            p.bob = 14.0
            p.lean = 14.0
        return p


def _phase(t: float, T: float) -> float:
    return 2.0 * math.pi * t / max(1e-6, T)


def generate_cycle(parts: List[dict], params: CycleParams) -> List[Pose]:
    """parts: list[dict] con chiavi did, tag, x, y, w, h (bbox full-image).

    Ritorna `params.frames` Pose; il frame N-1 fluisce nel frame 0 (loop).
    """
    # pivot per parte (calcolati una volta)
    pivots: Dict[str, Tuple] = {}
    base_center = [0.0, 0.0]
    tagged = 0
    for p in parts:
        pivots[p['did']] = pivot_for_tag(p.get('tag', ''), p['x'], p['y'], p['w'], p['h'])
        base_center[0] += p['x'] + p['w'] / 2.0
        base_center[1] += p['y'] + p['h'] / 2.0
        if p.get('tag'):
            tagged += 1
    base_center[0] /= max(1, len(parts))
    base_center[1] /= max(1, len(parts))

    kind = params.kind
    T = params.duration_s
    N = max(2, params.frames)

    def leg_phase(tag: str) -> float:
        # gamba destra fase 0, sinistra anti-fase
        return 0.0 if tag in ('legr',) else math.pi

    def arm_phase(tag: str) -> float:
        # braccio opposto alla gamba omonima
        if tag in ('armr', 'handr'):
            return math.pi  # braccio dx anti-fase con gamba dx
        return 0.0

    def body_group_dy(phi: float) -> float:
        # bob a DOPPIA frequenza (passo) per walk/run; nessun bob per idle
        # (il respiro e' gestito dal torso via scala)
        if kind == 'idle':
            return 0.0
        return params.bob * math.sin(2.0 * phi)

    frames: List[Pose] = []
    for i in range(N):
        t = i / N * T
        phi = _phase(t, T)
        pose = Pose()
        for p in parts:
            did, tag = p['did'], (p.get('tag') or '').lower()
            pp = PartPose(pivot=pivots[did])
            if kind == 'idle':
                if tag in TORSOTAGS:
                    pp.scale_y = 1.0 + params.breath * math.sin(2.0 * _phase(t, 3.0))
                    pp.scale_x = 1.0 - params.breath * 0.5 * math.sin(2.0 * _phase(t, 3.0))
                elif tag in HEAD_TAGS:
                    pp.angle = params.head_nod * math.sin(phi)
                    pp.dy = -0.6 * params.bob * math.sin(2.0 * _phase(t, 3.0))
                elif tag in ARM_TAGS:
                    pp.angle = params.arm_swing * math.sin(_phase(t, 3.0))
                elif tag in LEG_TAGS:
                    pp.angle = params.leg_swing * math.sin(_phase(t, 3.0))
            else:  # walk / run
                if tag in LEG_TAGS:
                    pp.angle = params.leg_swing * math.sin(phi + leg_phase(tag))
                    pp.dy = body_group_dy(phi)
                elif tag in ARM_TAGS:
                    pp.angle = params.arm_swing * math.sin(phi + arm_phase(tag))
                    pp.dy = body_group_dy(phi)
                elif tag in TORSOTAGS:
                    pp.angle = params.lean
                    pp.dy = body_group_dy(phi)
                elif tag in HEAD_TAGS:
                    pp.angle = params.lean * 0.25
                    pp.dy = body_group_dy(phi) * 0.6
                else:
                    pp.dy = body_group_dy(phi)
            pose[did] = pp
        frames.append(pose)
    return frames
