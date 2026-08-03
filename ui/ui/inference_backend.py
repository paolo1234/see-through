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

from .logger import logger as LOGGER
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

    def supports_batch(self): return True
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
        # generazione automatica leggera per CPU: niente multi-crop,
        # griglia 16x16, soglie basse, maschere <150px scartate
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        sam.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam.model,
            points_per_side=16,
            pred_iou_thresh=0.6,
            stability_score_thresh=0.7,
            min_mask_region_area=150,
            crop_n_layers=0,
        )
        return sam

    def infer_img(self, img, boxes=None, points=None, labels=None):
        if points is not None and len(points):
            pred_masks, scores, _ = self._model.predict(
                img, points=points, labels=labels)
        elif boxes is not None and len(boxes):
            pred_masks, scores, _ = self._model.predict(
                img, np.array(boxes))
        else:
            # batch automatico: segmenta tutto senza prompt
            return self._auto_segment(img)
        instances = []
        for m, s in zip(pred_masks, scores):
            m = np.array(m, dtype=bool)
            if not m.any():
                continue
            ys, xs = np.where(m)
            x, y = int(xs.min()), int(ys.min())
            w = int(xs.max()) - x + 1
            h = int(ys.max()) - y + 1
            # convenzione: Instance.mask e' CROP-LOCALE (bbox-relative)
            m = m[y:y + h, x:x + w]
            instances.append(Instance(mask=m.astype(np.uint8),
                                      bbox=[x, y, w, h],
                                      score=float(s), idx=len(instances)))
        return instances

    def _auto_segment(self, img):
        """Segmentazione automatica (nessun prompt): candidate maschere su tutta
        l'immagine. Su immagini grandi riduce il lato lungo a 1024px (la
        generazione automatica costa O(pixel^2)) e riporta le maschere a
        risoluzione originale."""
        from PIL import Image as PILImage
        h, w = img.shape[:2]
        scale = 1.0
        img_in = img
        max_side = 1024
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            img_in = np.array(PILImage.fromarray(img).resize((new_w, new_h)))
        try:
            raw = self._model.generate(img_in)
        except Exception:  # noqa: BLE001
            LOGGER.exception('Generazione automatica fallita; fallback a maschera piena')
            raw = []
        instances = []
        for r in raw:
            seg = np.asarray(r['segmentation'], dtype=bool)
            if int(r.get('area', 0)) < 100:
                continue
            if scale != 1.0:
                seg = np.asarray(
                    PILImage.fromarray(seg).resize((w, h), PILImage.Resampling.NEAREST),
                    dtype=bool)
            ys, xs = np.where(seg)
            if not ys.size:
                continue
            x, y = int(xs.min()), int(ys.min())
            bw = int(xs.max()) - x + 1
            bh = int(ys.max()) - y + 1
            m = seg[y:y + bh, x:x + bw]
            instances.append(Instance(mask=m.astype(np.uint8),
                                      bbox=[x, y, bw, bh],
                                      score=float(r.get('predicted_iou', 0.5)),
                                      idx=len(instances)))
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