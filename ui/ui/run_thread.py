import os
import os.path as osp

import numpy as np
from qtpy.QtCore import Signal
from PIL import Image

from .proj import ProjSeg
from .structures import Instance, save_instance_list
from .logger import logger as LOGGER
from .io_thread import ThreadBase


class SegmentationThread(ThreadBase):
    """Esegue l'inferenza (batch o box/point) tramite InferenceProvider.

    Segnali:
      page_finished(int)              - fine batch di una pagina
      manual_inference_finished(list) - nuove Instance prodotte da un prompt
      progress(int)                   - avanzamento 0..100
    """

    page_finished = Signal(int)
    manual_inference_finished = Signal(object)  # list[Instance]
    progress = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self._stop_flag = False

    def runSegmentation(self, proj: ProjSeg, provider, boxes=None, points=None, labels=None):
        """Avvia un job: batch (se senza prompt) o manuale (box/point)."""
        if self.job is not None:
            LOGGER.warning('SegmentationThread gia\' occupato; job scartato.')
            return
        if boxes is None and points is None:
            self.job = lambda: self._run_batch_seg(proj=proj, provider=provider)
        else:
            self.job = lambda: self._run_manual_inference(
                proj=proj, provider=provider, boxes=boxes,
                points=points, labels=labels)
        self.start()

    def _run_manual_inference(self, proj, provider, boxes, points, labels):
        img = proj.current_image
        if img is None:
            LOGGER.error('Nessuna immagine corrente per l\'inferenza manuale.')
            self.manual_inference_finished.emit([])
            return
        try:
            provider.load_model()  # lazy: carica anche nei path box/point
            new_instances = provider.infer_img(img, boxes=boxes, points=points, labels=labels)
        except Exception as e:  # noqa: BLE001
            LOGGER.exception('Inferenza manuale fallita')
            self.manual_inference_finished.emit([])
            raise e
        num_exists = len(proj.current_instance_list)
        for ii, ins in enumerate(new_instances):
            ins.idx = num_exists + ii
        # la persistenza/undo la gestisce il main thread via AddInstancesCommand
        self.manual_inference_finished.emit(list(new_instances))
        LOGGER.info(f'Inferenza manuale: +{len(new_instances)} istanze')

    def _run_batch_seg(self, proj: ProjSeg, provider):
        if not provider.supports_batch():
            LOGGER.error('Provider non supporta batch: usare box/point.')
            self.early_stop_signal.emit('Provider non supporta batch')
            return
        provider.load_model()
        for page_index, imgname in enumerate(proj.pages):
            if self._stop_flag and page_index > 0:
                self.early_stop_signal.emit('Batch interrotto dall\'utente')
                break
            page_dir = osp.join(proj.directory, imgname)
            if not osp.isdir(page_dir):
                continue
            from utils.io_utils import get_last_modified_file
            final = get_last_modified_file(osp.join(page_dir, 'final'),
                                           ['.jxl', '.png', '.webp'])
            if final is None:
                LOGGER.warning(f'final.* mancante in {page_dir}; skippo')
                continue
            img = np.array(Image.open(str(final)).convert('RGB'))
            try:
                instances = provider.infer_img(img)
            except Exception as e:  # noqa: BLE001
                LOGGER.exception(f'Batch fallito su {imgname}')
                continue
            for ii, ins in enumerate(instances):
                ins.idx = ii
            ins_path = proj.get_instance_path(imgname)
            os.makedirs(osp.dirname(ins_path), exist_ok=True)
            save_instance_list(instances, ins_path)
            self.page_finished.emit(page_index)
            self.progress.emit(int(round((page_index + 1) / max(1, proj.num_pages) * 100)))
        if not self._stop_flag:
            self.progress.emit(100)