"""Stage B substation segmentation on Esri Wayback z18 MBTiles, restricted to an AOI.

Copied from /mnt/drso-data/solar_lora/scripts/07_infer_corridor.py (original/07_infer_corridor.py,
sha256 in original/SHA256SUMS), which produced /mnt/cog-data/vivek/stageA/corridor_sub_prob_stageB.tif
with --ckpt .../stageB/lightning_logs/version_0/checkpoints/epoch22.ckpt. Kept identical:

  - 3072 px blocks with 128 px overlap on a grid ANCHORED AT THE MBTILES RASTER ORIGIN, visited in
    row-major order, so a later block overwrites the overlap of an earlier one exactly as in
    production (the in-AOI probabilities are bit-for-bit those of the corridor raster);
  - inside a block, 512 px tiles with 128 px overlap, probabilities averaged; edge tiles
    zero-padded; batch 24; fp16 autocast; softmax class 1;
  - ImageNet mean/std normalisation on RGB 0..255;
  - blocks whose alpha band is all 0 are skipped; prob = 0 where alpha = 0; uint8 prob*255,
    nodata 0, EPSG:3857 (the MBTiles grid).

Changes: AOI restriction (only blocks meeting `region` are run, prob = 0 outside it), output
window = region bounds instead of the whole archive, device/batch/paths as arguments, progress
and cancellation hooks.

Known, kept as production: training chips were UTM 0.60 m, inference runs on the native
mercator grid (0.597 m/px at the equator scale, ~0.548 m ground at 23.4 N). Not resampled,
because the delivered corridor results and thresholds were measured this way.
"""
from __future__ import annotations

import time

import numpy as np

MEANS = np.array([123.675, 116.28, 103.53], np.float32)
STDS = np.array([58.395, 57.12, 58.68], np.float32)
TILE, OV = 512, 128


def block_origins(H: int, W: int, block: int, ov: int = OV):
    """Production block grid over the full source raster (row-major)."""
    step = block - ov
    return [(by, bx) for by in range(0, H, step) for bx in range(0, W, step)]


def tile_offsets(bh: int, bw: int, tile: int = TILE, ov: int = OV):
    ts = tile - ov
    cys = list(range(0, max(bh - tile, 0) + 1, ts)) or [0]
    cxs = list(range(0, max(bw - tile, 0) + 1, ts)) or [0]
    if cys[-1] + tile < bh:
        cys.append(bh - tile)
    if cxs[-1] + tile < bw:
        cxs.append(bw - tile)
    return cys, cxs


def load_model(ckpt: str, device: str = "cuda"):
    """Caller must have put lulc_models/_shared on sys.path and set HF_HOME / HF_HUB_OFFLINE."""
    import custom_modules  # noqa: F401  (registers dinov3_hf_vitl16 with TerraTorch)
    from terratorch.tasks import SemanticSegmentationTask
    task = SemanticSegmentationTask.load_from_checkpoint(ckpt, map_location=device)
    task.eval().to(device)
    return task.model


def predict_block(model, img: np.ndarray, batch_size: int, device: str = "cuda") -> np.ndarray:
    """img (3, bh, bw) uint8 -> averaged class-1 probability (bh, bw) float32."""
    import torch
    _, bh, bw = img.shape
    prob = np.zeros((bh, bw), np.float32)
    cnt = np.zeros((bh, bw), np.float32)
    cys, cxs = tile_offsets(bh, bw)
    batch, pos = [], []

    def flush():
        if not batch:
            return
        t = torch.from_numpy(np.stack(batch)).to(device)
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = model(t)
            lg = out.output if hasattr(out, "output") else out
            p = torch.softmax(lg.float(), 1)[:, 1].cpu().numpy()
        for k, (yy, xx) in enumerate(pos):
            hh = min(TILE, bh - yy); ww = min(TILE, bw - xx)
            prob[yy:yy + hh, xx:xx + ww] += p[k, :hh, :ww]
            cnt[yy:yy + hh, xx:xx + ww] += 1
        batch.clear(); pos.clear()

    for yy in cys:
        for xx in cxs:
            c = img[:, yy:yy + TILE, xx:xx + TILE]
            if c.shape != (3, TILE, TILE):
                pad = np.zeros((3, TILE, TILE), img.dtype)
                pad[:, :c.shape[1], :c.shape[2]] = c
                c = pad
            n = (np.transpose(c, (1, 2, 0)).astype(np.float32) - MEANS) / STDS
            batch.append(np.transpose(n, (2, 0, 1)))
            pos.append((yy, xx))
            if len(batch) >= batch_size:
                flush()
    flush()
    prob /= np.maximum(cnt, 1)
    return prob


def run(src_path: str, region_3857, model, out_path: str, block: int = 3072, batch: int = 24,
        device: str = "cuda", progress=None, log=print) -> dict:
    """Run the model over the blocks of `src_path` that meet `region_3857`; write a uint8 COG-style
    GeoTIFF covering the region bounds. Returns stats."""
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds
    from shapely.geometry import box
    from shapely.prepared import prep

    src = rasterio.open(src_path)
    H, W = src.height, src.width
    full = Window(0, 0, W, H)
    ow = from_bounds(*region_3857.bounds, transform=src.transform)
    ow = ow.round_offsets(op="floor").round_lengths(op="ceil").intersection(full)
    oy, ox, oh, oww = int(ow.row_off), int(ow.col_off), int(ow.height), int(ow.width)
    prof = dict(driver="GTiff", height=oh, width=oww, count=1, dtype="uint8", crs=src.crs,
                transform=src.window_transform(ow), compress="deflate", tiled=True,
                blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER", nodata=0)
    preg = prep(region_3857)
    todo = []
    for by, bx in block_origins(H, W, block):
        bh = min(block, H - by); bw = min(block, W - bx)
        if bh < 64 or bw < 64:
            continue
        if by + bh <= oy or by >= oy + oh or bx + bw <= ox or bx >= ox + oww:
            continue
        bb = rasterio.windows.bounds(Window(bx, by, bw, bh), src.transform)
        if preg.intersects(box(*bb)):
            todo.append((by, bx, bh, bw))
    log("source %d x %d, output window %d x %d, blocks to run %d"
        % (W, H, oww, oh, len(todo)))

    t0 = time.time()
    skipped = 0
    px_pos = 0
    with rasterio.open(out_path, "w", **prof) as dst:
        for k, (by, bx, bh, bw) in enumerate(todo, 1):
            win = Window(bx, by, bw, bh)
            arr = src.read(window=win)
            if src.count >= 4:
                alpha = arr[3]
                if not alpha.any():
                    skipped += 1
                    continue
                valid = alpha > 0
            else:
                valid = np.ones((bh, bw), bool)
            valid &= geometry_mask([region_3857], out_shape=(bh, bw),
                                   transform=src.window_transform(win), invert=True)
            if not valid.any():
                skipped += 1
                continue
            prob = predict_block(model, arr[:3], batch, device)
            prob[~valid] = 0
            out8 = (prob * 255).astype(np.uint8)
            # clip the block to the output window
            y0, x0 = max(by, oy), max(bx, ox)
            y1, x1 = min(by + bh, oy + oh), min(bx + bw, ox + oww)
            part = out8[y0 - by:y1 - by, x0 - bx:x1 - bx]
            px_pos += int((part >= 128).sum())
            dst.write(part, 1, window=Window(x0 - ox, y0 - oy, x1 - x0, y1 - y0))
            if progress:
                el = time.time() - t0
                progress(k / len(todo), "block %d/%d (%d empty) ETA %.0f min"
                         % (k, len(todo), skipped, (len(todo) - k) * el / k / 60))
    src.close()
    return {"blocks": len(todo), "skipped": skipped, "px_ge_050": px_pos,
            "minutes": (time.time() - t0) / 60, "out": out_path}
