# -*- coding: utf-8 -*-
"""Flusso player reale: LOAD -> (istanze esistenti) -> APPLY -> ANIM -> EXPORT.
Niente dialoghi modali (offscreen safe)."""
import os, sys, time, tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from qtpy.QtWidgets import QApplication
app = QApplication([])
from ui import ui_config
ui_config.load_config()
from ui.proj import ProjSeg
from ui.anim.cycles import CycleParams, generate_cycle
from ui.anim.preview import render_pose
from ui.export.atlas import bake_atlas, save_atlas, scale_parts

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'project_player')
proj = ProjSeg()
t0 = time.perf_counter()
proj.load(os.path.join(base, 'proj.txt'))
print(f'LOAD {time.perf_counter()-t0:.1f}s pages={proj.num_pages}')
assert proj.model_valid
proj.set_current_page(proj.pages_index()[0] if hasattr(proj,'pages_index') else list(proj.pages)[0])
H, W = proj.current_image.shape[:2]
print(f'PAGE {W}x{H}')

# parti reali dai drawable
from ui.export.layers import collect_parts
parts = collect_parts(proj)
print(f'PARTI: {len(parts)} tags={[p["tag"] for p in parts][:8]}')
assert parts, 'nessuna parte'

# dialog: simula init + state change (senza modal)
from ui.anim_dialog import AnimationDialog
dlg = AnimationDialog(proj)
t0 = time.perf_counter()
dlg.state_combo.setCurrentText('run')
dlg._debounce.stop()
dlg._generate()
print(f'STATECHANGE+RIGENERA {time.perf_counter()-t0:.2f}s')
for kind in ('idle', 'walk', 'run'):
    p = dlg._last_params.get(kind) or CycleParams.defaults(kind)
    print(f'  {kind}: dur={p.duration_s} fr={p.frames} leg={p.leg_swing} arm={p.arm_swing}')

# export atlas 100% e 50%
for s in (1.0, 0.5):
    parts_s = scale_parts(parts, s)
    size = (int(round(W*s)), int(round(H*s)))
    cycles = {k: generate_cycle(parts_s, dlg._last_params.get(k) or CycleParams.defaults(k)) for k in ('idle','walk','run')}
    t0 = time.perf_counter()
    strip, manifest, report = bake_atlas(parts_s, cycles, size, padding=8)
    dt = time.perf_counter()-t0
    print(f'BAKE {int(s*100)}%: {dt:.1f}s sheet={strip.shape[1]}x{strip.shape[0]} cell={manifest["cellWidth"]}x{manifest["cellHeight"]} rows={list(manifest["rows"])}')
    out = tempfile.mkdtemp(prefix='st_p9_')
    f = save_atlas(strip, manifest, out)
    print(f'  saved {os.path.basename(f)} + manifest.json')

# persistenza
proj.save_anim({k: (dlg._last_params.get(k) or CycleParams.defaults(k)).to_dict() for k in ('idle','walk','run')})
assert os.path.exists(proj.anim_path())
loaded = proj.load_anim()
assert set(loaded) == {'idle','walk','run'}
dlg.close()
print('PLAYER FLOW OK (Fase 9)')
