# -*- coding: utf-8 -*-
# Fase 2 - Motorino inferenza a "provider" per la UI See-through.
#
# Astrae il backend di segmentazione dietro una interfaccia comune, cosi' la UI puo':
#   - girare in locale con SAM2.1 (torch moderno, gia' nel venv)   -> SAMProvider
#   - girare con il vecchio AnimeInsSeg/mmcv/mmdet (legacy)        -> CartoonSegProvider
#   - valutare il FLUSSO con un provider finto                    -> DummyProvider (test/dev)
#
# Ogni provider produce list[Instance] (vedi ui/ui/structures.py: mask, bbox=[x,y,w,h]).
import importlib
import numpy as np
import torch
from typing import List, Optional, Tuple

from .structures import Instance


class InferenceProvider(object):
    name: str = 'base'

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._model = None
        self._load_error = None

    # --- capacita' ---
    def supports_batch(self): return False
    def supports_box(self):   return False
    def supports_point(self): return False

    # --- caricamento lazy del modello ---
    def load_model(self):
        if self._model is None and self._load_error is None:
            try:
                self._model = self._build_model()
            except Exception as e:  # noqa: BLE001
                self._load_error = str(e)
                raise RuntimeError(
                    f"[{self.name}] modello non caricabile - {e}\n"
                    f"Controlla che dipendenze/pesi siano installati. "
                    f"Se serve solo il flusso, seleziona il provider 'dummy'."
                )

    def _build_model(self):
        raise NotImplementedError

    # definito dai sottoclassi; ritorna Instance
    def infer_img(self, img: np.ndarray, boxes=None, points=None, labels=None):
        raise NotImplementedError


class DummyProvider(InferenceProvider):
    """Provider finto: produce 1 bbox centrale per test/flusso senza GPU/pesi."""
    name = 'dummy'

    def supports_batch(self): return True
    def supports_box(self):   return True
    def supports_point(self): return not True  # solo box, per esercizio

    def _build_model(self):
        return {'dummy': True}

    def infer_img(self, img, boxes=None, points=None, labels=None):
        h, w = img.shape[:2]
        # un rettangolino "oggetto" centrato
        bw = max(8, w // 3)
        bh = max(8, h // 4)
        x, y = (w - bw) // 2, (h - bh) // 2
        mask = np.zeros((bh, bw), dtype=bool)
        mask[2:bh - 2, 2:bw - 2] = True
        return [Instance(mask=mask.astype(np.uint8),
                         bbox=[x, y, bw, bh],
                         score=0.99, idx=0)]


class SAMProvider(InferenceProvider):
    """Segment Anything 2.1 - funziona con torch moderno del nostro venv.
    Pesi scaricati da Meta alla prima chiamata (cache in HF_HOME).
    Su CPU usare model_size 'small' o 'tiny' (large = minuti per immagine)."""
    name = 'sam'
    weight_id = 'sam2.1_hiera_small'

    SIZES = {
        'tiny': 'sam2.1_hiera_tiny',
        'small': 'sam2.1_hiera_small',
        'base+': 'sam2.1_hiera_base_plus',
        'large': 'sam2.1_hiera_large',
    }

    def supports_box(self):   return True
    def supports_point(self): return True

    def _build_model(self):
        from annotators.lang_sam.models.sam import SAM
        size = self.kwargs.get('model_size', 'small')
        self.weight_id = self.SIZES.get(size, 'sam2.1_hiera_small')
        sam = SAM()
        device = self.kwargs.get('device', 'cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        sam.build_model(self.weight_id, device=device)
        return sam

    def infer_img(self, img, boxes=None, points=None, labels=None):
        if points is not None and len(points):
            pred_masks, scores, _ = self._model.predict(
                img, points=points, labels=labels)
        else:
            pred_masks, scores, _ = self._model.predict(
                img, np.array(boxes if boxes is not None else []))
        instances = []
        for m, s in zip(pred_masks, scores):
            m = np.array(m, dtype=bool)
            if not m.any():
                continue
            ys, xs = np.where(m)
            x, y = int(xs.min()), int(ys.min())
            w = int(xs.max()) - x + 1
            h = int(ys.max()) - y + 1
            instances.append(Instance(mask=m.astype(np.uint8),
                                      bbox=[x, y, w, h],
                                      score=float(s), idx=len(instances)))
        return instances


class CartoonSegProvider(InferenceProvider):
    """AnimeInsSeg (mmcv/mmdet legacy). Richiede install separato."""
    name = 'cartoonseg'

    def supports_batch(self): return True
    def supports_box(self):   return True

    def _build_model(self):
        # import pig per mescolinare conflitti col venv moderno
        return importlib.import_module(
            'annotators.animeinsseg.instance_segmentation')

    def batch_img(self, img, refine_method=None, instance_thresh=0.3):
        mod = self.load_model()
        rst = mod.apply_instance_segmentation(img, refine_method=refine_method,
                                              instance_thresh=instance_thresh)
        if rst.is_empty:
            return []
        instances = []
        for m, bb, sc in zip(rst.masks, rst.bboxes, rst.scores):
            instances.append(Instance(mask=m.astype(np.uint8), bbox=bb,
                                      score=sc, idx=len(instances)))
        return instances


_PROVIDERS = {}


def get_provider(name: str, **kwargs) -> InferenceProvider:
    """Ritorna un provider lancato, con errori chiari se il backend manca."""
    name = (name or 'dummy').lower()
    if name in _PROVIDERS:
        return _PROVIDERS[name]
    if name == 'dummy':
        provider = DummyProvider(**kwargs)
    elif name == 'sam':
        try:
            from .ui_config import pcfg
            kwargs.setdefault('model_size', getattr(pcfg, 'sam_model_size', 'small'))
        except Exception:  # noqa: BLE001
            pass
        provider = SAMProvider(**kwargs)
    elif name == 'cartoonseg':
        provider = CartoonSegProvider(**kwargs)
    else:
        raise ValueError(
            f"Provider '{name}' sconosciuto. Disponibili: "
            f"dummy, sam, cartoonseg.")
    _PROVIDERS[name] = provider
    return provider


AVAILABLE_PROVIDERS = ['dummy', 'sam', 'cartoonseg']