# -*- coding: utf-8 -*-
"""Dialogo "Cycle Animation": genera idle/walk/run parametrici dalle parti
applicate della pagina corrente, li mostra in preview animata e li esporta
come sprite atlas a righe + manifest.json (formato sprite-gen).
"""

import os

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (QDialog, QHBoxLayout, QVBoxLayout, QGridLayout,
                            QLabel, QPushButton, QComboBox, QDoubleSpinBox,
                            QSpinBox, QSlider, QFileDialog, QFrame)
from qtpy.QtGui import QPixmap

from .anim.cycles import CycleParams, generate_cycle
from .anim.preview import render_pose
from .anim.pose import Pose, PartPose, pivot_for_tag
from .export.atlas import bake_atlas, save_atlas
from .tagging import auto_tag, is_canonical, TAG_SCHEMA
from .logger import logger as LOGGER, create_info_dialog

STATES = ('idle', 'walk', 'run')


def _checker(size=(512, 512), cell=16):
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((xx // cell) + (yy // cell)) % 2
    base = np.where(m[..., None] > 0, np.array([46, 50, 58, 255]),
                    np.array([34, 37, 44, 255])).astype(np.uint8)
    return base


class AnimationDialog(QDialog):
    """Genera, rivede ed esporta cycle animation dalle parti applicate."""

    def __init__(self, proj, parent=None):
        super().__init__(parent)
        self.proj = proj
        self._parts = []
        self._cycles = {}
        self._current = 0
        self._playing = False
        self._busy = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        # debounce: una sola rigenerazione dopo la pausa (slider/spin drag)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(250)
        self._debounce.timeout.connect(self._generate)
        # parametri salvati per stato (anim.json della pagina): ricaricati
        # all'apertura e riscritti a chiusura/"Salva animazione"
        self._saved_params = {}
        if getattr(proj, 'model_valid', False):
            self._saved_params = proj.load_anim()
        self._last_params = {k: CycleParams.from_dict(v)
                             for k, v in self._saved_params.items()}
        self.setWindowTitle('Cycle Animation (idle / walk / run)')
        self.setMinimumSize(560, 620)
        self._build_ui()
        self.state_combo.blockSignals(True)
        self.state_combo.setCurrentText('walk')  # stato iniziale: walk
        self.state_combo.blockSignals(False)
        self._generate()

    # ---------------- UI ----------------
    def _build_ui(self):
        lay = QVBoxLayout(self)

        self.preview = QLabel(self)
        self.preview.setFixedSize(512, 512)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet('border: 1px solid #3a3d44; border-radius: 4px;')
        lay.addWidget(self.preview, 1)

        # slider + contatore frame
        row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_slider)
        self.frame_label = QLabel('0/8', self)
        row.addWidget(self.frame_slider, 1)
        row.addWidget(self.frame_label)
        lay.addLayout(row)

        # controlli
        grid = QGridLayout()
        r = 0
        self.state_combo = QComboBox(self)
        self.state_combo.addItems(list(STATES))
        self.state_combo.currentTextChanged.connect(self._on_state_changed)
        grid.addWidget(QLabel('Stato:', self), r, 0)
        grid.addWidget(self.state_combo, r, 1)

        self.duration_spin = QDoubleSpinBox(self)
        self.duration_spin.setRange(0.2, 10.0)
        self.duration_spin.setSingleStep(0.1)
        self.duration_spin.setDecimals(1)
        self.duration_spin.valueChanged.connect(self._on_param_changed)
        grid.addWidget(QLabel('Durata (s):', self), r, 2)
        grid.addWidget(self.duration_spin, r, 3)

        r += 1
        self.frames_spin = QSpinBox(self)
        self.frames_spin.setRange(4, 24)
        self.frames_spin.valueChanged.connect(self._on_param_changed)
        grid.addWidget(QLabel('Frame:', self), r, 0)
        grid.addWidget(self.frames_spin, r, 1)

        self.leg_spin = QDoubleSpinBox(self)
        self.leg_spin.setRange(0, 90)
        self.leg_spin.valueChanged.connect(self._on_param_changed)
        grid.addWidget(QLabel('Gambe (deg):', self), r, 2)
        grid.addWidget(self.leg_spin, r, 3)

        r += 1
        self.arm_spin = QDoubleSpinBox(self)
        self.arm_spin.setRange(0, 90)
        self.arm_spin.valueChanged.connect(self._on_param_changed)
        grid.addWidget(QLabel('Braccia (deg):', self), r, 0)
        grid.addWidget(self.arm_spin, r, 1)

        self.bob_spin = QDoubleSpinBox(self)
        self.bob_spin.setRange(0, 40)
        self.bob_spin.valueChanged.connect(self._on_param_changed)
        grid.addWidget(QLabel('Bob (px):', self), r, 2)
        grid.addWidget(self.bob_spin, r, 3)

        r += 1
        self.res_combo = QComboBox(self)
        self.res_combo.addItems(['100%', '75%', '50%', '25%'])
        self.res_combo.setToolTip('Risoluzione di esportazione (default: nativa)')
        grid.addWidget(QLabel('Risoluzione:', self), r, 0)
        grid.addWidget(self.res_combo, r, 1)

        r += 1
        self.lean_spin = QDoubleSpinBox(self)
        self.lean_spin.setRange(0, 30)
        self.lean_spin.valueChanged.connect(self._on_param_changed)
        grid.addWidget(QLabel('Lean (deg):', self), r, 0)
        grid.addWidget(self.lean_spin, r, 1)
        lay.addLayout(grid)

        self.parts_label = QLabel(self)
        self.parts_label.setWordWrap(True)
        lay.addWidget(self.parts_label)

        # pulsanti
        btns = QHBoxLayout()
        self.play_btn = QPushButton(self.tr('Genera & preview'), self)
        self.play_btn.clicked.connect(self._generate)
        btns.addWidget(self.play_btn)
        self.playpause_btn = QPushButton(self.tr('▶ Play'), self)
        self.playpause_btn.clicked.connect(self._toggle_play)
        btns.addWidget(self.playpause_btn)
        self.stop_btn = QPushButton(self.tr('⏹ Stop'), self)
        self.stop_btn.clicked.connect(self._stop)
        btns.addWidget(self.stop_btn)
        self.save_btn = QPushButton(self.tr('Salva animazione'), self)
        self.save_btn.setToolTip('Persiste i parametri per stato in anim.json ')
        self.save_btn.clicked.connect(self._save_anim)
        btns.addWidget(self.save_btn)
        self.export_btn = QPushButton(self.tr('Esporta atlas…'), self)
        self.export_btn.clicked.connect(self._export)
        btns.addWidget(self.export_btn)
        btns.addStretch(1)
        lay.addLayout(btns)

        # riga export aggiuntiva: layers PNG / PSD
        lay2 = QHBoxLayout()
        self.layers_png_btn = QPushButton(self.tr('Esporta layers PNG…'), self)
        self.layers_png_btn.clicked.connect(self._export_layers_png)
        lay2.addWidget(self.layers_png_btn)
        self.layers_psd_btn = QPushButton(self.tr('Esporta PSD…'), self)
        self.layers_psd_btn.clicked.connect(self._export_layers_psd)
        lay2.addWidget(self.layers_psd_btn)
        self.seq_btn = QPushButton(self.tr('Sequenze PNG…'), self)
        self.seq_btn.clicked.connect(self._export_sequences)
        lay2.addWidget(self.seq_btn)
        lay2.addStretch(1)
        lay.addLayout(lay2)

        self._apply_defaults('walk')

    # ---------------- dati ----------------
    def _load_parts(self):
        if not self.proj.model_valid or self.proj.current_image is None:
            create_info_dialog('Apri una pagina con parti applicate (Apply as parts).')
            return []
        H, W = self.proj.current_image.shape[:2]
        parts = []
        for dr in self.proj.l2dmodel.valid_drawables():
            try:
                crop = np.asarray(dr.get_img(), dtype=np.uint8)
            except Exception as e:  # noqa: BLE001
                LOGGER.warning(f'anim: skip {dr.did}: {e}')
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
            # esclude le maschere quasi-intera immagine (background di SAM)
            if area > 0.55 * H * W:
                continue
            parts.append({'did': dr.did, 'tag': (dr.tag or '').lower(),
                          'x': int(getattr(dr, 'x', 0)), 'y': int(getattr(dr, 'y', 0)),
                          'w': w, 'h': h, 'area': area,
                          'img': crop})
        if not parts:
            create_info_dialog('Nessuna parte applicata: applica prima i candidati '
                               'con "Apply as parts".')
            return []
        # tag canonici (mantiene quelli gia' validi, stima gli altri)
        tags = auto_tag(parts, (H, W))
        for p in parts:
            p['tag'] = tags.get(p['did'], 'unknown')
        return parts

    def _params(self):
        p = CycleParams(kind=self.state_combo.currentText(),
                        duration_s=float(self.duration_spin.value()),
                        frames=int(self.frames_spin.value()),
                        leg_swing=float(self.leg_spin.value()),
                        arm_swing=float(self.arm_spin.value()),
                        bob=float(self.bob_spin.value()),
                        lean=float(self.lean_spin.value()))
        if p.kind == 'idle':
            p.leg_swing = min(p.leg_swing, 4.0)
            p.arm_swing = min(p.arm_swing, 4.0)
            p.lean = 0.0
        return p

    def _apply_defaults(self, kind):
        # usa i parametri salvati/persistiti se presenti, altrimenti i default
        p = self._last_params.get(kind) or CycleParams.defaults(kind)
        self._busy = True
        try:
            self.duration_spin.setValue(p.duration_s)
            self.frames_spin.setValue(p.frames)
            self.leg_spin.setValue(p.leg_swing)
            self.arm_spin.setValue(p.arm_swing)
            self.bob_spin.setValue(p.bob)
            self.lean_spin.setValue(p.lean)
        finally:
            self._busy = False

    # ---------------- generazione ----------------
    def _generate(self):
        if self._busy:
            return
        parts = self._load_parts()
        if not parts:
            return
        self._parts = parts
        kind = self.state_combo.currentText()
        p = self._params()
        self._last_params[kind] = p
        self._cycles = {kind: generate_cycle(parts, p)}
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(1, len(self._cycles[kind]) - 1))
        self.frame_slider.blockSignals(False)
        self._current = 0
        n_known = sum(1 for x in parts if x['tag'] in ('head', 'torso', 'chest',
                                                       'waist', 'hip', 'skirt',
                                                       'arml', 'armr', 'handl',
                                                       'handr', 'legl', 'legr',
                                                       'footl', 'footr'))
        self.parts_label.setText(
            f'Parti: {len(parts)} ({n_known} animate, {len(parts) - n_known} statiche) — '
            + ', '.join(p['tag'] for p in parts[:10]) + ('…' if len(parts) > 10 else ''))
        self._render_current()
        self._stop()

    def _render_current(self):
        if not self._cycles:
            return
        kind = self.state_combo.currentText()
        frames = self._cycles.get(kind)
        if not frames:
            return
        i = min(self._current, len(frames) - 1)
        H, W = self.proj.current_image.shape[:2]
        img = render_pose(self._parts, frames[i], (W, H))
        # checker sotto per vedere l'alpha
        checker = _checker((H, W))
        a = img[..., 3:4].astype(np.float64) / 255.0
        out = (img[..., :3].astype(np.float64) * a
               + checker[..., :3].astype(np.float64) * (1 - a)).astype(np.uint8)
        qimg = np.concatenate([out, np.full(out.shape[:2] + (1,), 255, np.uint8)], axis=-1)
        from .misc import ndarray2pixmap
        pix = ndarray2pixmap(qimg)
        pix = pix.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(pix)
        self.frame_label.setText(f'{i + 1}/{len(frames)}')

    # ---------------- playback ----------------
    def _toggle_play(self):
        if self._playing:
            self._stop()
        else:
            self._play()

    def _play(self):
        if not self._cycles:
            return
        p = self._params()
        interval = max(30, int(round(p.duration_s / max(1, p.frames) * 1000)))
        self._playing = True
        self.playpause_btn.setText(self.tr('⏸ Pausa'))
        self._timer.start(interval)

    def _stop(self):
        self._playing = False
        self._timer.stop()
        self.playpause_btn.setText(self.tr('▶ Play'))

    def _next_frame(self):
        kind = self.state_combo.currentText()
        n = len(self._cycles.get(kind, []))
        if n <= 1:
            self._stop()
            return
        self._current = (self._current + 1) % n
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self._current)
        self.frame_slider.blockSignals(False)
        self._render_current()

    def _on_slider(self, v):
        self._current = v
        self._render_current()

    def _on_state_changed(self, kind):
        if self._busy:
            return
        self._apply_defaults(kind)
        self._debounce.start()

    def _on_param_changed(self, _=None):
        if self._busy:
            return
        self._debounce.start()  # rigenera una sola volta dopo la pausa

    # ---------------- persistenza ----------------
    def _save_anim(self):
        """Serializza i parametri effettivi dei 3 stati in anim.json."""
        if not getattr(self.proj, 'model_valid', False):
            create_info_dialog('Progetto non aperto: impossibile salvare.')
            return
        out = {}
        for kind in STATES:
            p = self._last_params.get(kind) or CycleParams.defaults(kind)
            out[kind] = p.to_dict()
        self.proj.save_anim(out)
        self._saved_params = out
        create_info_dialog(
            f'Animazione salvata in:\n{self.proj.anim_path()}\n\n'
            f'Parametri per stato ({len(out)}): durata, frame, swing gambe/'
            'braccia, bob, lean. Saranno ricaricati alla prossima apertura.')

    def closeEvent(self, event):
        # salvataggio automatico (silenzioso) dei parametri al termine
        if self._last_params and getattr(self.proj, 'model_valid', False):
            try:
                out = {k: (self._last_params.get(k) or CycleParams.defaults(k)).to_dict()
                       for k in STATES}
                self.proj.save_anim(out)
            except Exception as e:  # noqa: BLE001
                LOGGER.warning(f'anim persist: {e}')
        super().closeEvent(event)

    # ---------------- export ----------------
    def _export_layers_png(self):
        from .export.layers import export_layers_png, collect_parts
        parts = collect_parts(self.proj)
        if not parts:
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, self.tr('Esporta layers PNG…'),
            self.proj.instance_dir() if hasattr(self.proj, 'instance_dir') else '')
        if not out_dir:
            return
        files = export_layers_png(self.proj, parts, out_dir)
        create_info_dialog(
            f'Esportati {len(files)} file in:\n{out_dir}\n\n'
            'background.png = pagina originale\n'
            'preview.png = composita parti+sfondo\n'
            '<NN>_<tag>_<did>.png = singola parte (alpha).')

    def _export_layers_psd(self):
        from .export.layers import export_layers_psd, collect_parts
        parts = collect_parts(self.proj)
        if not parts:
            return
        default = os.path.join(
            self.proj.instance_dir() if hasattr(self.proj, 'instance_dir') else '',
            f'{self.proj.current_model or "layers"}.psd')
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr('Esporta PSD…'), default, 'Photoshop (*.psd)')
        if not path:
            return
        if not path.lower().endswith('.psd'):
            path += '.psd'
        n = export_layers_psd(self.proj, parts, path)
        create_info_dialog(f'PSD con {n} layer salvato in:\n{path}')

    def _export_sequences(self):
        """PNG sequence per stato (idle/walk/run) dai parametri correnti."""
        if not self._parts:
            create_info_dialog('Nessuna parte da esportare.')
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, self.tr('Esporta sequenze PNG…'),
            self.proj.instance_dir() if hasattr(self.proj, 'instance_dir') else '')
        if not out_dir:
            return
        from .export.atlas import scale_parts
        from .export.sequences import export_sequences
        H, W = self.proj.current_image.shape[:2]
        res_txt = self.res_combo.currentText()
        scale = float(res_txt.rstrip('%')) / 100.0
        parts = scale_parts(self._parts, scale)
        size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))
        cycles = {}
        for kind in STATES:
            p = self._last_params.get(kind) or CycleParams.defaults(kind)
            cycles[kind] = generate_cycle(parts, p)
        n = len(export_sequences(parts, cycles, size, out_dir))
        create_info_dialog(f'Esportate {n} sequenze in:\n{out_dir}')

    def _export(self):
        if not self._parts:
            create_info_dialog('Nessuna parte da esportare.')
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, self.tr('Esporta atlas animato…'),
            self.proj.instance_dir() if hasattr(self.proj, 'instance_dir') else '')
        if not out_dir:
            return
        H, W = self.proj.current_image.shape[:2]
        res_txt = self.res_combo.currentText()
        scale = float(res_txt.rstrip('%')) / 100.0
        self.export_btn.setText(self.tr('Esportazione in corso…'))
        self.export_btn.setEnabled(False)
        from .export.atlas import scale_parts
        parts = scale_parts(self._parts, scale)
        size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))
        cycles = {}
        for kind in STATES:
            p = self._last_params.get(kind) or CycleParams.defaults(kind)
            cycles[kind] = generate_cycle(parts, p)
        try:
            strip, manifest, report = bake_atlas(parts, cycles, size, padding=8)
            sheet = save_atlas(strip, manifest, out_dir)
        except ValueError as e:
            create_info_dialog(str(e))
        finally:
            self.export_btn.setText(self.tr('Esporta atlas…'))
            self.export_btn.setEnabled(True)
        create_info_dialog(
            f'Atlas esportato in:\n{sheet}\n\n'
            f'manifest.json con frame_layout ({len(STATES)} stati, '
            f'cella {manifest["cellWidth"]}x{manifest["cellHeight"]}, '
            f'risoluzione {res_txt}).\n'
            f'Compatibile col formato sprite-gen.')
        LOGGER.info(f'Atlas animato: {sheet}')
