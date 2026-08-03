# 🎮 Asset Pipeline 2D — "See-Through Studio" → Game-Ready

> Visione: un'unica pipeline (desktop, CPU-friendly) che porta un'immagine qualsiasi —
> disegnata, scaricata o **generata da AI** — fino a **personaggi animati pronti per il gioco**:
> sprite atlas, rigs scheletrici, cycle animation, colorways, export Godot/Spine/PNG.
>
> Ispirazione: **stretchyStudio** (l'engine di animazione ufficiale del modello See-Through:
> auto-rigging, mesh-deformation "stretchy", shape keys, export Spine 4), **Spine**
> (standard industriale: ossa, IK, skinning, curve), **Live2D** (deformazione mesh + parametri),
> **Godot** (AnimationPlayer/Skeleton2D gratis e leggero), **sprite-gen** (skill già installata:
> atlases a strisce per stato + palette swap deterministico).

---

## 0. La pipeline in un colpo d'occhio

```
 [0] CREAZIONE          [1] DECOMPOSIZIONE     [2] RIGGING          [3] ANIMAZIONE
 ─────────────────      ─────────────────      ─────────────────    ─────────────────
 sprite-gen CLI    →    SAM auto/batch     →   auto-skeleton    →   timeline/keyframe
 (base char, frames)    edit maschere          (euristiche o         cycle generator
 import PNG/PSD         tagging semantico      DWPose onnx CPU)     (walk/idle/run)
                        inpaint parti          ossa + skinning       mesh deformation
                        ordine profondità      pivot/joint           shape keys (blink)
                              │                     │                     │
                              └──────────┬──────────┘─────────────────────┘
                                         ▼
                              [4] EXPORT GAME-READY
                              ─────────────────────
                              • sprite atlas strips + manifest.json  (formato sprite-gen)
                              • Godot .tscn/.tres (AnimationPlayer o AnimatedSprite2D)
                              • Spine 4 JSON
                              • PNG sequence per stato
                              • colourway bake (sprite-gen recolor) a partire da 1 palette
                                         ▼
                              [5] QA / AUDIT (atlas check, rig check, cycle check)
```

Ogni stadio produce **file JSON/PNG** aperti e versionabili → stesso formato consumabile
dall'app desktop, da Colab/Gradio, e dai motori di gioco. Niente formati proprietari.

---

## 1. Stage 0 — Creazione asset (AI)

**Obiettivo**: produrre o importare l'immagine base del personaggio.

| Tool | Stato | Note |
|---|---|---|
| `sprite-gen` CLI (skill installata) | ✅ esistente | genera character base, frame, row strips, colourway bake (`recolor`); pipeline component-row |
| Import PNG/PSD | ✅ esistente (UI) | drag&drop nella UI |
| `sprite-gen gen` multiprovider | ✅ esistente | codex/grok/registry provider |

**Da fare (leggero)**:
- pulsante "Genera personaggio" nella UI → subprocess `sprite-gen` con i parametri UI.
- import `.psd` (già supportato dal repo lato `see-through`; la UI usa PNG).

---

## 2. Stage 1 — Decomposizione in layer (il "See-Through" locale)

**Obiettivo**: separare il personaggio in parti semantiche **complete** (testa, busto, braccio
sx/dx, gamba sx/dx...) con maschera + alpha, pronte per rigging.

| Capacità | Stato |
|---|---|
| Segmentazione batch SAM (auto) | ✅ fatto (13 candidati / 512² ~27s CPU) |
| Segmentazione box/point SAM | ✅ fatto |
| Edit maschere (pennello, split, merge) | ✅ fatto |
| Apply as parts + undo + persistenza | ✅ fatto |
| Export layered PNG + manifest | ✅ fatto |
| Tagging semantico parti | ✅ fatto (euristiche geometriche, `ui/ui/tagging.py`, tag canonici lowercase, esclusione background) |
| **Inpaint delle parti occluse** | ⚠️ da pianificare: è il cuore del paper (layerdiff3d, GPU). Su CPU: fallback "extend from edges" per MVP, vero inpaint su Colab (script `inference_psd.py` già esistente) |
| Ordinamento profondità (z-order) | ⚠️ parziale: ordine drawables; il paper usa depth (marigold). MVP: euristiche + override manuale |

**Da fare (track 1)**: tagging semantico (schema tag standard
`head, torso, armL, armR, legL, legR, handL, handR, hairF, hairB, skirt, ...`) con
normalizzazione: è la base per l'auto-skeleton e per il cycle generator.

---

## 3. Stage 2 — Rigging

**Obiettivo**: scheletro + parenting parti→ossa + skinning + pivot.

### 3.1 Auto-skeleton (CPU)
- **Euristica** (MVP, istantanea): dai tag + bbox delle parti →
  - testa = circle sopra il collo; torso = rettangolo centrale; pelvi sotto; arti = segmenti dal giunto (spalla→gomito→mano, anca→ginocchio→piede) stimati da bbox
  - gerarchia: `root → hips → spine → chest → neck → head`, `chest → shoulderL → elbowL → handL` ...
- **DWPose via onnxruntime CPU** (opzionale, 1–3 s per 512²): scheletro vero, keypoints COCO-17 → mappa sulle parti. onnx modello `dw-ll_ucoco_384.onnx` (~230 MB) — fattibile anche su questo PC.

### 3.2 Editor ossa (manuale)
- aggiungere/spostare osso, ri-parentare, impostare pivot (joint anchor) per parte.
- **Skinning**: ogni parte ha una maschera; weights = distanza trasforma→osso (heat diffusion) o pennello manuale. Ogni vertice della mesh della parte riceve peso 0..1 per osso.

### 3.3 Mesh per deformazione "stretchy" (Live2D-style)
- per parti "morbide" (capelli, gonna, guance): griglia mesh generata dal bbox della maschera + rilassamento sul bordo alpha; warp barycentrico affine per vertice.
- **limb bending**: gomiti/ginocchia con skinning a 2 ossa + correzione "bend" (mantieni spessore, stile Live2D).

**Da fare (track 2)** — moduli `ui/ui/rig/`.

---

## 4. Stage 3 — Animazione

### 4.1 Timeline editor (modulo `ui/ui/anim/`)
- Clip (idle, walk, run, attack...), track (osso: pos/rot/scale; parte: visibilità, opacity, z-order; mesh: vertici; shape keys), keyframe con interpolazione (linear / ease-in-out / bezier), onion skin, loop preview.
- Reuse dell'esistente: canvas scene graph + QUndoStack + `updateCanvas()`.

### 4.2 Cycle animation generator (MVP ad altissimo valore, CPU-puro)
Generatore **parametrico** di cycle animation senza bisogno di animare a mano:
- **Walk cycle**: seno con fase sfasata per gamba/braccio opposti, foot "plant" con IK 2-bone, bob verticale del corpo = 2× frequenza passo, leggero rollio bacino/spalle.
- **Idle**: respiro (scala torso/chest ±1.5%, fase 3s), blink shape-key ogni 2.5–4s random.
- **Run**: come walk con frequenza/ampiezza maggiori e airborne pose.
- Parametri esposti in UI: durata ciclo (default 1.0s walk), passo, ampiezza braccia, ecc.
- Output: bake in **N frame per stato** → sprite strip (riga per stato) + manifest frame_layout — **stesso formato di sprite-gen** → i motori lo consumano identico.

### 4.3 Mesh deformation / shape keys
- warp mesh su vertici (capelli/gonna che oscillano col ciclo), shape keys (blink, sorriso) con slider d'influenza — come stretchyStudio/Blender.

**Da fare (track 3)**: timeline piena; (track 3b) cycle generator.

---

## 5. Stage 4 — Export game-ready

| Formato | Uso | Stato |
|---|---|---|
| **Atlas strips + manifest.json** | sprite sheet frame-based (Godot AnimatedSprite2D, Unity, Defold, Phaser) — stesso schema `frame_layout` di sprite-gen | ✅ fatto (baker `ui/ui/export/atlas.py` + dialogo Anim) |
| **Layered PNG per parte** | `background.png` + `preview.png` + `NN_tag_did.png` (alpha) — base per rigging in qualsiasi engine | ✅ fatto (`ui/ui/export/layers.py`, bottoni nel dialogo Anim) |
| **PSD con un layer per parte** | Photoshop/Clip Studio/After Effects (layer RGBA + offset pagina) | ✅ fatto (psd-tools, `export_layers_psd`) |
| **Godot .tscn/.tres** | Skeleton2D + AnimationPlayer con keyframe; o AnimatedSprite2D+SpriteFrames | ❌ da fare |
| **Spine 4 JSON** | standard industriale (Spine/DragonBones/engine vari) | ❌ da fare |
| PNG sequence per stato | pipeline generiche | ❌ da fare (banale, dal baker) |
| **Colourway bake** | 1 personaggio + palette map → N colorways in un comando (`sprite-gen recolor`) | ✅ CLI esiste; da integrare in UI |

---

## 6. Stage 5 — QA / Audit

- **Atlas check**: frame fuori bounds, buchi alpha, overdraw, padding mancante.
- **Rig check**: parti non parentate, ossa NaN, weight non normalizzati, skinning fuori immagine.
- **Cycle check**: continuità primo/ultimo frame (loop seamless), foot sliding, differenza delta-frame.
- Report in pannello Audit della UI (fase 7 del piano originale) + export JSON.

---

## 7. Architettura (estensione dell'app esistente)

```
ui/ui/
├── rig/
│   ├── bones.py          # Bone: id, parent, pos, rot, scale (local+world)
│   ├── skeleton.py       # Skeleton: gerarchia, auto-build da tag/bbox, salvataggio rig.json
│   ├── skinning.py       # weights per parte (heat/penna), apply = transform per osso
│   └── mesh.py           # griglia mesh per parte + warp barycentrico
├── anim/
│   ├── timeline.py       # widget timeline (playhead, track, keyframe)
│   ├── clips.py          # Clip/Track/Keyframe dataclasses + interpolazione (bezier)
│   ├── sampler.py        # valuta clip a tempo t → pose (dict osso→transform)
│   ├── cycles.py         # walk/idle/run parametrici (sine + IK 2-bone) + serializzazione CycleParams
│   └── preview.py        # render pose su canvas (warp deterministico via cv2.remap)
├── export/
│   ├── atlas.py          # bake pose→frames→strip + manifest frame_layout (formato sprite-gen)
│   ├── layers.py         # layered PNG per parte + PSD (psd-tools) — Fase 9
│   ├── godot.py          # .tscn/.tres (Skeleton2D+AnimationPlayer | AnimatedSprite2D)
│   ├── spine.py          # Spine 4 JSON (bones/slots/skins/animations)
│   └── sequences.py      # PNG sequence per stato
├── settings_dialog.py    # provider/device + parametri SAM batch (persistiti in ui_config)
├── help_dialog.py        # F1: scorciatoie + guida flusso
└── mainwindow.py         # + pannelli Rig / Timeline / Export
```

Tutto resta **file-based JSON** (stesso spirito di `instances.json` / `manifest.json`):
`rig.json`, `clips.json`, `frames.json`, `anim.json` (parametri cycle per stato,
salvati/ricaricati dal dialogo Anim) — versionabili, leggibili da Colab/Gradio.

---

## 8. Roadmap implementativa (ordine consigliato)

| Track | Cosa | Valore | Costo | Stato |
|---|---|---|---|---|
| **1** | Tagging semantico parti + schema tag standard | abilita tutto il resto | S | ✅ |
| **2** | **Cycle generator + atlas baker** (walk/idle/run → strip + manifest) | 💥 deliverable immediato: sprite animati pronti per il gioco senza timeline | M | ✅ (bottone "Anim" nella TopArea; dialog con preview/play/export, risoluzione 25–100%, persistenza parametri in `anim.json`, export layers PNG/PSD) |
| **8** | **Completezza UI desktop (Fase 8–9)**: layer ops undo-able, canvas interattivo, clipboard/shortcut, Settings (SAM batch), help F1, did2drawable sync, warp deterministico, fix crash | robustezza quotidiana | M | ✅ |
| **3** | Rigging: auto-skeleton euristico + editor ossa + skinning | abilita animazione scheletrica | M–L | |
| **4** | Timeline editor completo + export Godot/Spine | produzione professionale | L | |
| **5** | Mesh deformation + shape keys (stretchy) | livello Live2D | L | |
| **6** | DWPose onnx (auto-skeleton di qualità) | migliora rigging automatico | M | |
| **7** | Inpaint parti su Colab (layerdiff) + sync UI | parti veramente complete (occlusioni) | M (GPU) | |

**Prossimo passo consigliato: Track 3** (rigging: auto-skeleton euristico da tag/bbox +
editor ossa) — abilita l'animazione scheletrica e l'export Godot/Spine.
