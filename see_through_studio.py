"""
See-through Studio — GUI Gradio per la pipeline See-through.
Stage: LayerDiff (layer RGBA) -> Depth (Marigold) -> editing alpha -> inpaint LaMa ->
ricomposizione custom (selezione/ordine) -> rigenerazione guidata del part -> export PSD.

Eseguire dentro la cartella <repo>/see-through.
Su Colab:
    import see_through_studio as st
    st.app.queue().launch(share=True)

Tutti i .png intermedi stanno in <REPO>/workspace/studio/<nome>/. Lo stato (paths,
sorgente) vive in gr.State e si aggiorna a ogni run.
"""
import os, sys, shutil, subprocess
from pathlib import Path

REPO = os.environ.get("SEE_THROUGH_REPO", "/content/see-through")
for p in (REPO, os.path.join(REPO, "common"), os.path.join(REPO, "annotators")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

def _ensure_pil():
    # Pillow su Colab può essere in stato misto (ImageText presente ma PIL._typing senza _Ink).
    # Verifica in un sottoprocesso separato per non inquinare il cache di questo processo.
    probe = "import sys; from PIL import _typing; sys.exit(0 if getattr(_typing, '_Ink', None) is not None else 3)"
    if subprocess.run([sys.executable, "-c", probe]).returncode != 0:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                        "--force-reinstall", "--no-cache-dir", "Pillow>=11.1.0"], check=False)

_ensure_pil()
from PIL import Image, ImageDraw
import cv2
import torch
import gradio as gr

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def checker(h, w, s=8):
    b = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(b)
    for y in range(0, h, s):
        for x in range(0, w, s):
            if (x // s + y // s) % 2 == 0:
                d.rectangle([x, y, x + s - 1, y + s - 1], fill=(200, 200, 200))
    return b

def rgba_on_checker(arr, maxside=420):
    img = Image.fromarray(arr[..., :4].copy())
    img.thumbnail((maxside, maxside))
    bg = checker(*img.size).convert("RGBA")
    bg.alpha_composite(img)
    return np.asarray(bg.convert("RGB"))

def list_part_pngs(namedir):
    out = []
    if not namedir or not Path(namedir).is_dir():
        return []
    for pp in sorted(Path(namedir).glob("*.png")):
        if pp.stem in ("src_img", "src_head", "reconstruction") or pp.stem.endswith("_depth"):
            continue
        out.append(pp.stem)
    return out

def load_preview(namedir_or_path, tag=None):
    p = Path(namedir_or_path) if tag is None else Path(namedir_or_path) / ("%s.png" % tag)
    if not p.exists():
        return None
    return rgba_on_checker(np.array(Image.open(p).convert("RGBA")))

def read_layer(namedir, tag):
    p = Path(namedir) / ("%s.png" % tag)
    if not p.exists():
        return None
    return np.array(Image.open(p).convert("RGBA"))

def split_order(txt):
    if not txt:
        return []
    return [x.strip() for x in txt.split(",") if x.strip()]

# ---------------------------------------------------------------------------
# pipeline wrappers (lazy)
# ---------------------------------------------------------------------------
def run_layerdiff(srcp, saved_dir, repo, res, steps, seed, offload):
    from utils.inference_utils import apply_layerdiff
    return apply_layerdiff(srcp, repo, save_dir=saved_dir, resolution=res,
                           num_inference_steps=steps, seed=seed,
                           group_offload=offload, disable_progressbar=True)

def run_depth(srcp, saved_dir, repo, res, steps, seed, offload):
    from utils.inference_utils import apply_marigold
    return apply_marigold(srcp, repo, resolution=res, num_inference_steps=steps,
                          seed=seed, save_dir=saved_dir, group_offload=offload,
                          disable_progressbar=True)

def run_assemble(namedir):
    from utils.inference_utils import further_extr
    further_extr(namedir, rotate=False, save_to_psd=True, tblr_split=False)

# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------
def process(src, repo_id, depth_repo, resolution, res_depth, steps, depth_steps,
            seed, offload):
    src = getattr(src, "name", src)
    if not src or not os.path.exists(src):
        raise gr.Error("Carica prima un'immagine")

    save_dir = os.path.join(os.getcwd(), "workspace", "studio")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    name = Path(src).stem
    namedir = os.path.join(save_dir, name)
    Path(namedir).mkdir(parents=True, exist_ok=True)

    logs = []
    try:
        logs.append("Step 1/3: LayerDiff...")
        run_layerdiff(src, save_dir, repo_id, int(resolution), int(steps), int(seed), bool(offload))
        logs.append("  layer RGBA ok")
    except Exception as e:
        logs.append("  ERRORE LayerDiff: %r" % (e,))

    try:
        logs.append("Step 2/3: Depth...")
        run_depth(src, save_dir, depth_repo, int(res_depth), int(depth_steps), int(seed), bool(offload))
        logs.append("  depth ok")
    except Exception as e:
        logs.append("  ERRORE Depth: %r" % (e,))

    parts = list_part_pngs(namedir)
    gal = [(load_preview(namedir, t), t) for t in parts]
    depths = sorted(Path(namedir).glob("*_depth.png"))
    dgal = [(load_preview(str(d)), d.stem) for d in depths]

    psd_path = None
    try:
        logs.append("Step 3/3: PSD...")
        run_assemble(namedir)
        psds = list(Path(save_dir).rglob("*.psd"))
        if psds:
            psd_path = str(max(psds, key=lambda p: p.stat().st_mtime))
        logs.append("  PSD: %s" % (os.path.basename(psd_path) if psd_path else "niente",))
    except Exception as e:
        logs.append("  ERRORE PSD: %r" % (e,))

    _u = gr.update(choices=parts, value=parts[0] if parts else None)
    state = {"namedir": namedir, "src": src, "name": name, "save_dir": save_dir}
    order = ",".join(parts)
    return "\n".join(logs), gal, dgal, psd_path, state, _u, _u, gr.update(value=order)

# ---------------------------------------------------------------------------
# editor alpha (singolo layer)
# ---------------------------------------------------------------------------
def show_part(state, tag):
    img = None if not state else read_layer(state["namedir"], tag)
    return None if img is None else rgba_on_checker(img)

def _edit_alpha(a, thr, erode, dilate, blur):
    if blur > 0:
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=float(blur))
    if thr > 0:
        a = np.where(a >= int(thr), 255, a).astype(np.uint8)
    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(erode) * 2 + 1, int(erode) * 2 + 1))
        a = cv2.erode(a, k)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dilate) * 2 + 1, int(dilate) * 2 + 1))
        a = cv2.dilate(a, k)
    return a

def edit_preview(state, tag, thr, erode, dilate, blur):
    img = None if not state else read_layer(state["namedir"], tag)
    if img is None:
        return None
    img = img.copy()
    img[..., 3] = _edit_alpha(img[..., 3].copy(), thr, erode, dilate, blur)
    return rgba_on_checker(img)

def edit_commit(state, tag, thr, erode, dilate, blur):
    img = None if not state else read_layer(state["namedir"], tag)
    if img is None:
        return None
    img = img.copy()
    img[..., 3] = _edit_alpha(img[..., 3].copy(), thr, erode, dilate, blur)
    Image.fromarray(img).save(Path(state["namedir"]) / ("%s.png" % tag))
    return rgba_on_checker(img)

# ---------------------------------------------------------------------------
# rigenerazione guidata del singolo layer (re-run diffusion + sostituzione)
# ---------------------------------------------------------------------------
def regen_part(state, src, tag, repo_id, resolution, steps, seed, offload):
    if not state or not tag:
        return None
    name = state["name"]
    tmp = Path(state["save_dir"]) / ("_regen_" + name)
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        run_layerdiff(src, str(tmp), repo_id, int(resolution), int(steps), int(seed), bool(offload))
        srcf = tmp / name / ("%s.png" % tag)
        if srcf.exists():
            shutil.copy(str(srcf), os.path.join(state["namedir"], "%s.png" % tag))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return load_preview(state["namedir"], tag)

# ---------------------------------------------------------------------------
# inpaint su zona (LaMa)
# ---------------------------------------------------------------------------
def load_inp_base(state, tag):
    if not state or not tag:
        return None
    return read_layer(state["namedir"], tag)[..., :3] if read_layer(state["namedir"], tag) is not None else None

def mask_from_paint(base, painted):
    if isinstance(painted, (tuple, list)) and len(painted) == 2:
        painted = painted[0]
    painted = np.asarray(painted, dtype=np.float32)
    h, w = painted.shape[:2]
    b = base
    if b.shape[:2] != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    diff = np.abs(painted - b.astype(np.float32)).max(axis=2)
    return (diff > 32).astype(np.uint8) * 255, b

def inpaint_run(state, tag, painted, save):
    if not state or not tag:
        return None
    base = load_inp_base(state, tag)
    if base is None or painted is None:
        return None
    mask, b = mask_from_paint(base, painted)
    layer = read_layer(state["namedir"], tag)
    alpha = np.zeros((0, 0), np.uint8)
    if layer is not None:
        alpha = layer[..., 3]
    if alpha.shape[:2] != b.shape[:2]:
        alpha = cv2.resize(alpha, (b.shape[1], b.shape[0]))
    out = b.copy()
    if mask.sum() > 0:
        try:
            from annotators.lama_inpainter import apply_inpaint
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            out = apply_inpaint(b, mask, device=dev)
            out = np.clip(np.round(out), 0, 255).astype(np.uint8)
        except Exception:
            out = cv2.inpaint(b, mask, 3, cv2.INPAINT_TELEA)
    full = np.concatenate([out, alpha], axis=2).astype(np.uint8)
    if save:
        Image.fromarray(full).save(Path(state["namedir"]) / ("%s.png" % tag))
    return rgba_on_checker(full)

def inpaint_preview(state, tag, painted):
    return inpaint_run(state, tag, painted, save=False)

def inpaint_commit(state, tag, painted):
    return inpaint_run(state, tag, painted, save=True)

# ---------------------------------------------------------------------------
# selezione / ordine / ricomposizione
# ---------------------------------------------------------------------------
def refresh_parts(state):
    if not state:
        return gr.update(choices=[], value=[])
    pr = list_part_pngs(state["namedir"])
    return gr.update(choices=pr, value=pr)

def pick_order(state):
    if not state:
        return gr.update(choices=[], value=None)
    pr = list_part_pngs(state["namedir"])
    return gr.update(choices=pr, value=pr[0] if pr else None)

def auto_order(state, sel):
    if not state:
        return None
    sel = sel or []
    ranks = []
    for t in sel:
        dp = Path(state["namedir"]) / ("%s_depth.png" % t)
        m = 0.0
        if dp.exists():
            a = np.array(Image.open(dp).convert("L"))
            m = float(a[a > 0].mean()) if (a > 0).any() else 0.0
        ranks.append((m, t))
    ranks.sort(reverse=True)
    return ",".join(t for _, t in ranks)

def mode_rowtxt(txt, sel_name, delta):
    parts = split_order(txt)
    if sel_name in parts:
        i = parts.index(sel_name)
        j = i + delta
        if 0 <= j < len(parts):
            parts[i], parts[j] = parts[j], parts[i]
    return ",".join(parts)

def move_up(order_txt, sel_name):
    return mode_rowtxt(order_txt, sel_name, -1)

def move_down(order_txt, sel_name):
    return mode_rowtxt(order_txt, sel_name, 1)

def recompose(state, sel, order_txt):
    if not state:
        return None
    sel = sel or []
    order = split_order(order_txt)
    if not order:
        order = list(sel)
    canvas = None
    for t in order:
        fp = Path(state["namedir"]) / ("%s.png" % t)
        if not fp.exists():
            continue
        layer = np.array(Image.open(fp).convert("RGBA"))
        if canvas is None:
            canvas = Image.new("RGBA", (layer.shape[1], layer.shape[0]), (0, 0, 0, 0))
        canvas.alpha_composite(Image.fromarray(layer))
    if canvas is None:
        return None
    return rgba_on_checker(np.asarray(canvas))

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build():
    with gr.Blocks(title="See-through Studio") as app:
        gr.Markdown("## 🎛 See-through Studio")
        st = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1) Input & Config")
                src = gr.File(label="Immagine", file_types=["image"])
                repo_id = gr.Dropdown(
                    choices=["layerdifforg/seethroughv0.0.2_layerdiff3d",
                             "24yearsold/seethroughv0.0.1_layerdiff3d"],
                    value="layerdifforg/seethroughv0.0.2_layerdiff3d", label="repo layer")
                resolution = gr.Slider(768, 1280, value=1024, step=64, label="res layer")
                steps = gr.Slider(8, 40, value=30, step=1, label="step diffusion")
                depth_repo = gr.Textbox(value="24yearsold/seethroughv0.0.1_marigold", label="repo depth")
                res_depth = gr.Slider(512, 768, value=768, step=64, label="res depth")
                depth_steps = gr.Slider(1, 16, value=4, step=1, label="step depth")
                seed = gr.Number(value=42, precision=0, label="seed")
                offload = gr.Checkbox(value=True, label="group_offload")
                run_btn = gr.Button("Run pipeline", variant="primary")
                log = gr.Textbox(label="Log", lines=10)
            with gr.Column(scale=2):
                gr.Markdown("### Layer RGBA")
                gal = gr.Gallery(label="Layer", columns=4, height=420, object_fit="contain")
                gr.Markdown("### Depth per layer")
                dgal = gr.Gallery(label="Depth", columns=4, height=260)
                psd_out = gr.File(label="PSD")

        gr.Markdown("---\n### 2) Editor del singolo layer")
        with gr.Row():
            with gr.Column(scale=1):
                part_tag = gr.Dropdown(label="Parte", choices=[], value=None)
                thr = gr.Slider(0, 255, value=60, step=5, label="alpha threshold")
                erode = gr.Slider(0, 30, value=0, step=1, label="erode (px)")
                dilate = gr.Slider(0, 30, value=0, step=1, label="dilate (px)")
                blur = gr.Slider(0, 10, value=0, step=1, label="gauss blur")
                prev_btn = gr.Button("Anteprima (alpha)", variant="secondary")
                comm_btn = gr.Button("Commit alpha", variant="primary")
                gr.Markdown("**Rigenera via diffusion** (refrun LayerDiff, salva solo questo layer)")
                regen_btn = gr.Button("♻️ Rigenera questo layer (diff)", variant="huggingface")
            with gr.Column(scale=2):
                part_view = gr.Image(label="Editor parte", type="numpy", height=420)

        gr.Markdown("### 3) Ricomposizione custom (scegli + ordina)")
        with gr.Row():
            with gr.Column():
                sel = gr.Dropdown(multiselect=True, label="Parti incluse", choices=[], value=[])
                order_sel = gr.Dropdown(label="Scegli cosa spostare", choices=[], value=None)
                order_txt = gr.Textbox(label="Ordine (virgola, alto prima)", placeholder="head,face,front hair,...")
                up_btn = gr.Button("↑ Su", variant="secondary")
                down_btn = gr.Button("↓ Giù", variant="secondary")
                auto_btn = gr.Button("✨ Auto-ordine per profondità", variant="secondary")
                ref_btn = gr.Button("Aggiorna elenco", variant="secondary")
                rec_btn = gr.Button("Ricomponi", variant="primary")
            with gr.Column():
                rec_view = gr.Image(label="Ricomposizione", type="numpy", height=420)

        gr.Markdown("### 4) Inpaint su zona (LaMa)")
        with gr.Row():
            with gr.Column(scale=1):
                inp_part = gr.Dropdown(label="Parte da inpaintare", choices=[], value=None)
                inp_sketch = gr.Image(
                    label="1) scegli la parte (si carica) · 2) disegna la zona rossa · 3) inpaint",
                    tool="sketch", type="numpy", height=360)
                inp_prev = gr.Button("Anteprima inpaint", variant="secondary")
                inp_comm = gr.Button("Commit inpaint", variant="primary")
            with gr.Column(scale=2):
                inp_view = gr.Image(label="Risultato inpaint", type="numpy", height=420)

        gr.Markdown("---\n*I .png intermedi stanno in `workspace/studio/<nome>/`; tutte le modifiche scrivono lì (sincronizzate).*")

        # ----------------------- eventi -----------------------
        part_tag.change(fn=show_part, inputs=[st, part_tag], outputs=part_view)
        prev_btn.click(fn=edit_preview, inputs=[st, part_tag, thr, erode, dilate, blur], outputs=part_view)
        comm_btn.click(fn=edit_commit, inputs=[st, part_tag, thr, erode, dilate, blur], outputs=part_view)
        regen_btn.click(fn=regen_part, inputs=[st, src, part_tag, repo_id, resolution, steps, seed, offload], outputs=part_view)

        ref_btn.click(fn=refresh_parts, inputs=[st], outputs=sel)
        order_sel.click(fn=pick_order, inputs=[st], outputs=order_sel)
        up_btn.click(fn=move_up, inputs=[order_txt, order_sel], outputs=order_txt)
        down_btn.click(fn=move_down, inputs=[order_txt, order_sel], outputs=order_txt)
        auto_btn.click(fn=auto_order, inputs=[st, sel], outputs=order_txt)
        rec_btn.click(fn=recompose, inputs=[st, sel, order_txt], outputs=rec_view)

        inp_part.change(fn=load_inp_base, inputs=[st, inp_part], outputs=inp_sketch)
        inp_prev.click(fn=inpaint_preview, inputs=[st, inp_part, inp_sketch], outputs=inp_view)
        inp_comm.click(fn=inpaint_commit, inputs=[st, inp_part, inp_sketch], outputs=inp_view)

        run_btn.click(fn=process, inputs=[src, repo_id, depth_repo, resolution, res_depth,
                                          steps, depth_steps, seed, offload],
                      outputs=[log, gal, dgal, psd_out, st, part_tag, inp_part, order_txt])

    return app


app = build()

if __name__ == "__main__":
    app.queue().launch(share=True)