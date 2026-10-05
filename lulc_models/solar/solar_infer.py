"""Solar segmentation inference over Esri Wayback z18 MBTiles, restricted to the AOI.

Port of /mnt/drso-data/solar_lora/scripts/07_infer_corridor.py. Unchanged: 3072 px blocks with
128 px block overlap on the MBTiles' own pixel grid (row-major, later blocks overwrite the overlap),
512 px chips with 128 px overlap averaged, ImageNet mean/std on RGB, fp16 autocast, softmax class 1,
alpha == 0 pixels zeroed, uint8 prob*255 EPSG:3857 output.
Changed: only blocks intersecting the AOI are run, pixels outside the AOI are zeroed, the output
raster covers just the AOI bbox window, chips with no valid pixel are skipped (exact: a chip only
contributes to its own pixels), weights/HF cache come from the request, no network.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "_shared"
MEANS = np.array([123.675, 116.28, 103.53], np.float32)
STDS = np.array([58.395, 57.12, 58.68], np.float32)


def set_offline_env(hf_home: str | None):
    """Must run before transformers is imported."""
    if hf_home:
        os.environ["HF_HOME"] = str(hf_home)
        os.environ.pop("HF_HUB_CACHE", None)
        os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def import_custom_modules():
    """Registers dinov3_hf_vitl16 in the TerraTorch backbone registry."""
    if str(SHARED) not in sys.path:
        sys.path.insert(0, str(SHARED))
    import custom_modules  # noqa: F401
    return custom_modules


def load_model(checkpoint: str, hf_home: str | None, device: str):
    set_offline_env(hf_home)
    import_custom_modules()
    from terratorch.tasks import SemanticSegmentationTask
    task = SemanticSegmentationTask.load_from_checkpoint(checkpoint, map_location=device)
    task.eval().to(device)
    return task.model


def aoi_window(src, aoi_3857) -> Window | None:
    full = Window(0, 0, src.width, src.height)
    w = from_bounds(*aoi_3857.bounds, transform=src.transform)
    w = w.round_offsets(op="floor").round_lengths(op="ceil")
    try:
        w = w.intersection(full)
    except Exception:  # rasterio raises WindowError when disjoint
        return None
    if w.width < 1 or w.height < 1:
        return None
    return Window(int(w.col_off), int(w.row_off), int(w.width), int(w.height))


def run(mbtiles: str, aoi_3857, out_tif: str, model, device: str = "cuda", block: int = 3072,
        tile: int = 512, overlap: int = 128, batch_size: int = 24, progress=None, log=print) -> dict:
    """Writes out_tif (uint8 prob*255) over the AOI window. Returns stats, or {} if no overlap."""
    import torch

    stats = {"blocks": 0, "blocks_run": 0, "blocks_skipped": 0, "pos_px": 0}
    with rasterio.open(mbtiles) as src:
        aw = aoi_window(src, aoi_3857)
        if aw is None:
            return {}
        W, H = src.width, src.height
        a_c0, a_r0 = aw.col_off, aw.row_off
        a_c1, a_r1 = a_c0 + aw.width, a_r0 + aw.height
        prof = dict(driver="GTiff", height=aw.height, width=aw.width, count=1, dtype="uint8",
                    crs=src.crs, transform=src.window_transform(aw), compress="deflate",
                    tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER", nodata=0)
        step = block - overlap
        ys = [y for y in range(0, H, step) if y < a_r1 and y + block > a_r0]
        xs = [x for x in range(0, W, step) if x < a_c1 and x + block > a_c0]
        total = len(ys) * len(xs)
        stats["blocks"] = total
        t0 = time.time()
        done = 0
        with rasterio.open(out_tif, "w", **prof) as dst:
            for by in ys:
                for bx in xs:
                    done += 1
                    bh, bw = min(block, H - by), min(block, W - bx)
                    if bh < 64 or bw < 64:
                        continue
                    win = Window(bx, by, bw, bh)
                    in_aoi = ~geometry_mask([aoi_3857], out_shape=(bh, bw),
                                            transform=src.window_transform(win))
                    if not in_aoi.any():
                        stats["blocks_skipped"] += 1
                        continue
                    arr = src.read(window=win)
                    valid = (arr[3] > 0) if src.count >= 4 else np.ones((bh, bw), bool)
                    valid &= in_aoi
                    if not valid.any():
                        stats["blocks_skipped"] += 1
                        continue
                    prob = _predict_block(arr[:3], valid, model, device, tile, overlap, batch_size, torch)
                    prob[~valid] = 0
                    out8 = (prob * 255).astype(np.uint8)
                    # write the part of this block inside the AOI window
                    c0, r0 = max(bx, a_c0), max(by, a_r0)
                    c1, r1 = min(bx + bw, a_c1), min(by + bh, a_r1)
                    sub = out8[r0 - by:r1 - by, c0 - bx:c1 - bx]
                    dst.write(sub, 1, window=Window(c0 - a_c0, r0 - a_r0, c1 - c0, r1 - r0))
                    stats["blocks_run"] += 1
                    stats["pos_px"] += int((sub >= 153).sum())
                    if progress:
                        progress(done / total, f"inference block {done}/{total}")
                    if done % 25 == 0:
                        el = time.time() - t0
                        log(f"[{done}/{total}] run={stats['blocks_run']} {el / 60:.1f} min "
                            f"ETA {(total - done) * el / max(done, 1) / 60:.0f} min")
    stats["window"] = [aw.col_off, aw.row_off, aw.width, aw.height]
    return stats


def _predict_block(img, valid, model, device, tile, overlap, batch_size, torch):
    _, bh, bw = img.shape
    prob = np.zeros((bh, bw), np.float32)
    cnt = np.zeros((bh, bw), np.float32)
    ts = tile - overlap
    cys = list(range(0, max(bh - tile, 0) + 1, ts)) or [0]
    cxs = list(range(0, max(bw - tile, 0) + 1, ts)) or [0]
    if cys[-1] + tile < bh:
        cys.append(bh - tile)
    if cxs[-1] + tile < bw:
        cxs.append(bw - tile)
    batch, pos = [], []
    use_amp = str(device).startswith("cuda")

    def flush():
        if not batch:
            return
        t = torch.from_numpy(np.stack(batch)).to(device)
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=use_amp):
            out = model(t)
            lg = out.output if hasattr(out, "output") else out
            p = torch.softmax(lg.float(), 1)[:, 1].cpu().numpy()
        for k, (yy, xx) in enumerate(pos):
            hh, ww = min(tile, bh - yy), min(tile, bw - xx)
            prob[yy:yy + hh, xx:xx + ww] += p[k, :hh, :ww]
            cnt[yy:yy + hh, xx:xx + ww] += 1
        batch.clear()
        pos.clear()

    for yy in cys:
        for xx in cxs:
            if not valid[yy:yy + tile, xx:xx + tile].any():
                continue
            c = img[:, yy:yy + tile, xx:xx + tile]
            if c.shape != (3, tile, tile):
                pad = np.zeros((3, tile, tile), img.dtype)
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


def crop_prob(prob_tif: str, aoi_3857, out_tif: str, rows: int = 4096) -> dict:
    """Dev/test stand-in for run(): copy the AOI window of an existing uint8 prob raster
    (same 3857 grid convention), zeroing pixels outside the AOI. Streams row bands."""
    with rasterio.open(prob_tif) as src:
        aw = aoi_window(src, aoi_3857)
        if aw is None:
            return {}
        tr = src.window_transform(aw)
        prof = dict(driver="GTiff", height=aw.height, width=aw.width, count=1, dtype="uint8",
                    crs=src.crs, transform=tr, compress="deflate", tiled=True, blockxsize=512,
                    blockysize=512, BIGTIFF="IF_SAFER", nodata=0)
        with rasterio.open(out_tif, "w", **prof) as dst:
            for y in range(0, aw.height, rows):
                h = min(rows, aw.height - y)
                arr = src.read(1, window=Window(aw.col_off, aw.row_off + y, aw.width, h))
                sub = Window(0, y, aw.width, h)
                arr[geometry_mask([aoi_3857], out_shape=arr.shape, transform=dst.window_transform(sub))] = 0
                dst.write(arr, 1, window=sub)
    return {"window": [aw.col_off, aw.row_off, aw.width, aw.height], "source": prob_tif}
