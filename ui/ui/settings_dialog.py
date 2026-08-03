"""Dialog Impostazioni: provider, modello SAM, parametri batch, risoluzione.

Fase 8: espone in UI i parametri prima hardcodati (inference_backend) e
li persiste nel config utente (ui_config.save_config).
"""
from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                            QComboBox, QDoubleSpinBox, QSpinBox, QPushButton,
                            QLabel, QCheckBox, QGroupBox)
from qtpy.QtCore import Qt

from .ui_config import pcfg, save_config
from .inference_backend import SAMProvider, AVAILABLE_PROVIDERS
from .logger import logger as LOGGER


class SettingsDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr('Impostazioni'))
        self.setMinimumWidth(380)
        lay = QVBoxLayout(self)

        # ---- Inferenza ----
        box = QGroupBox(self.tr('Inferenza'), self)
        form = QFormLayout(box)

        self.provider_combo = QComboBox(box)
        self.provider_combo.addItems(AVAILABLE_PROVIDERS)
        self.provider_combo.setCurrentText(pcfg.inference_provider)
        form.addRow(self.tr('Engine:'), self.provider_combo)

        self.size_combo = QComboBox(box)
        self.size_combo.addItems(list(SAMProvider.SIZES.keys()))
        self.size_combo.setCurrentText(pcfg.sam_model_size)
        self.size_combo.setToolTip(self.tr('Modello SAM2.1 (CPU: usa tiny o small)'))
        form.addRow(self.tr('Modello SAM:'), self.size_combo)

        self.device_combo = QComboBox(box)
        self.device_combo.addItems(['cpu', 'cuda'])
        self.device_combo.setCurrentText(pcfg.segmentation_device)
        form.addRow(self.tr('Device:'), self.device_combo)
        lay.addWidget(box)

        # ---- Parametri batch SAM ----
        box2 = QGroupBox(self.tr('Parametri batch SAM (segment-everything)'), self)
        form2 = QFormLayout(box2)

        self.points_spin = QSpinBox(box2)
        self.points_spin.setRange(4, 64)
        self.points_spin.setValue(int(pcfg.sam_points_per_side))
        form2.addRow(self.tr('Griglia punti (points_per_side):'), self.points_spin)

        self.iou_spin = QDoubleSpinBox(box2)
        self.iou_spin.setRange(0.1, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(float(pcfg.sam_pred_iou_thresh))
        form2.addRow(self.tr('Soglia IoU predetto:'), self.iou_spin)

        self.stab_spin = QDoubleSpinBox(box2)
        self.stab_spin.setRange(0.1, 1.0)
        self.stab_spin.setSingleStep(0.05)
        self.stab_spin.setValue(float(pcfg.sam_stability_score_thresh))
        form2.addRow(self.tr('Soglia stabilità:'), self.stab_spin)

        self.area_spin = QSpinBox(box2)
        self.area_spin.setRange(0, 5000)
        self.area_spin.setSingleStep(25)
        self.area_spin.setValue(int(pcfg.sam_min_mask_region_area))
        form2.addRow(self.tr('Area minima maschera (px):'), self.area_spin)

        self.side_spin = QSpinBox(box2)
        self.side_spin.setRange(256, 4096)
        self.side_spin.setSingleStep(128)
        self.side_spin.setValue(int(pcfg.sam_max_batch_side))
        self.side_spin.setToolTip(self.tr('Lato lungo massimo usato per il batch su immagini grandi'))
        form2.addRow(self.tr('Lato max batch (px):'), self.side_spin)
        lay.addWidget(box2)

        # ---- pulsanti ----
        btns = QHBoxLayout()
        btns.addStretch(1)
        save_btn = QPushButton(self.tr('Salva'), self)
        save_btn.clicked.connect(self._apply)
        cancel_btn = QPushButton(self.tr('Annulla'), self)
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        btns.addWidget(save_btn)
        lay.addLayout(btns)

    def _apply(self):
        pcfg.inference_provider = self.provider_combo.currentText()
        pcfg.sam_model_size = self.size_combo.currentText()
        pcfg.segmentation_device = self.device_combo.currentText()
        pcfg.sam_points_per_side = int(self.points_spin.value())
        pcfg.sam_pred_iou_thresh = float(self.iou_spin.value())
        pcfg.sam_stability_score_thresh = float(self.stab_spin.value())
        pcfg.sam_min_mask_region_area = int(self.area_spin.value())
        pcfg.sam_max_batch_side = int(self.side_spin.value())
        save_config()
        LOGGER.info('Impostazioni salvate (provider=%s, sam=%s, device=%s)',
                    pcfg.inference_provider, pcfg.sam_model_size, pcfg.segmentation_device)
        self.accept()
