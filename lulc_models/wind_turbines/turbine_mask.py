"""Full-scene round4 turbine probability mask (GPU).

Copied from /mnt/drso-data/devesh/turbine_expg/build_mask_edge.py, with the helpers it imported
from /home/vivek_jain/seggpt-local/turbines_work/run_turbine_full_archive.py (MEANS, STDS, CHIP,
scene_wide_stretch_bounds, make_windows) inlined verbatim.

Windows with valid_frac >= 0.99 are inferred as-is; windows with 0.05 <= valid_frac < 0.99 have
their no-data filled with noise resampled from the window's own valid pixels. Predictions are
kept on valid pixels only; overlapping windows merge by elementwise max. Chip 896 px at the
native 0.45 m, stride 672, batch 16, fp32 softmax class 1 -> uint8 on the scene grid.

Run on the WHOLE scene: the 1-99 percentile stretch is scene-wide and Exp-G calibrates on the
whole scene's blob statistics.

Changes vs the original: checkpoint / device / VRAM cap / output path are arguments (the
GPU_MEM_FRACTION=0.4 shared-box constant is replaced by the worker's vram_mb cap); a GSD
mismatch raises GSDError (caller skips the scene) instead of an assert; a cancel/progress hook
between windows. Model is loaded once per run (load_model) and reused across scenes.
"""
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.join(HERE, "..", "_shared")

# --- from run_turbine_full_archive.py (L58-62, L81-106) ------------------------------------
MEANS = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STDS = np.array([58.395, 57.12, 58.68], dtype=np.float32)
CHIP = 896


def scene_wide_stretch_bounds(src, overview_factor=8):
    h, w = src.height, src.width
    decim = src.read(1, out_shape=(h // overview_factor, w // overview_factor))
    valid = decim[decim > 0]
    lo, hi = np.percentile(valid, [1, 99])
    return float(lo), float(hi)


def make_windows(w, h, stride):
    windows = []
    ny = max(1, -(-(h - CHIP) // stride) + 1) if h > CHIP else 1
    nx = max(1, -(-(w - CHIP) // stride) + 1) if w > CHIP else 1
    for iy in range(ny):
        y = min(iy * stride, max(0, h - CHIP))
        for ix in range(nx):
            x = min(ix * stride, max(0, w - CHIP))
            windows.append(Window(x, y, CHIP, CHIP))
    # de-dup windows clamped to the same edge position
    seen = set()
    uniq = []
    for win in windows:
        key = (win.col_off, win.row_off)
        if key not in seen:
            seen.add(key)
            uniq.append(win)
    return uniq
# -------------------------------------------------------------------------------------------

STRIDE = 672
BATCH = 16
MIN_FRAC = 0.05


class GSDError(ValueError):
    pass


def load_model(checkpoint, vram_mb=None):
    """Load the DINOv3-LoRA round4 SemanticSegmentationTask onto cuda:0 (CUDA_VISIBLE_DEVICES is
    pinned by common.gpu_env). Needs HF_HOME pointing at a cache holding
    facebook/dinov3-vitl16-pretrain-lvd1689m and HF_HUB_OFFLINE=1."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if SHARED not in sys.path:
        sys.path.insert(0, SHARED)
    import torch
    import custom_modules  # noqa: F401  registers dinov3_hf_vitl16 in the terratorch registry
    from terratorch.tasks import SemanticSegmentationTask
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available: the turbine mask step needs a GPU")
    if vram_mb:
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(min(0.95, vram_mb / total), device=0)
    return SemanticSegmentationTask.load_from_checkpoint(checkpoint, map_location="cuda").eval().to("cuda").model


def build_mask(model, raster, dst, gsd_m=0.45, gsd_tol=0.01, on_progress=None, log=print):
    """Write <dst> (uint8 class-1 probability * 255 on the scene grid). Returns stats dict."""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(0)
    t0 = time.time()
    with rasterio.open(raster) as src:
        W, H = src.width, src.height; gsd = abs(src.res[0])
        if abs(gsd - gsd_m) >= gsd_tol:
            raise GSDError(f"unexpected GSD {gsd}")
        lo, hi = scene_wide_stretch_bounds(src)
        mosaic = np.zeros((H, W), np.uint8); batch = []; n_full = n_edge = n_skip = 0

        def flush():
            if not batch: return
            x = (np.stack([a for a, _, _ in batch]).astype(np.float32) - MEANS[None, :, None, None]) / STDS[None, :, None, None]
            with torch.no_grad():
                p = F.softmax(model(torch.from_numpy(x).cuda()).output, dim=1)[:, 1].cpu().numpy()
            for (_, w, valid), pr in zip(batch, p):
                u8 = np.clip(pr * 255, 0, 255).astype(np.uint8)
                if valid is not None: u8[~valid] = 0
                sl = mosaic[w.row_off:w.row_off + CHIP, w.col_off:w.col_off + CHIP]; np.maximum(sl, u8, out=sl)
            batch.clear()

        wins = make_windows(W, H, STRIDE)
        for i, w in enumerate(wins):
            band = src.read(1, window=w); valid = band > 0; f = valid.mean()
            if f < MIN_FRAC: n_skip += 1; continue
            s = np.clip((band.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
            if f < 0.99:
                s[~valid] = rng.choice(s[valid], size=(~valid).sum()); n_edge += 1
            else:
                valid = None; n_full += 1
            u8 = (s * 255).astype(np.uint8); batch.append((np.stack([u8, u8, u8]), w, valid))
            if len(batch) == BATCH: flush()
            if on_progress and (i + 1) % 200 == 0:
                on_progress((i + 1) / len(wins))
            if (i + 1) % 1000 == 0: log(f"  {i+1}/{len(wins)} windows ({time.time()-t0:.0f}s)")
        flush()
        prof = src.profile; prof.update(driver="GTiff", count=1, dtype="uint8", nodata=None, compress="deflate", predictor=2,
                                        tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    with rasterio.open(dst + ".tmp.tif", "w", **prof) as d: d.write(mosaic, 1)
    os.replace(dst + ".tmp.tif", dst)
    st = dict(full=n_full, edge=n_edge, skipped=n_skip, seconds=round(time.time() - t0))
    log(f"mask {os.path.basename(dst)}: full {n_full} + edge {n_edge} windows, {n_skip} empty skipped, {st['seconds']}s")
    return st
