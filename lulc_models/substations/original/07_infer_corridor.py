#!/usr/bin/env python3
"""Full-corridor inference straight off the Wayback MBTiles.

The holdout eval only covered 14.91 km2 (0.30% of the AOI) -- tight crops around
7 known OSM sites. That measures segmentation where solar is known to exist and
says nothing about false positives across the rest of the corridor. This runs
the whole 4,909 km2.

Legitimate as a holdout for SOLAR: training sites were national with the
corridor cut out by geometry, so none of this ground was seen.

Reads RGBA windows from the MBTiles; the alpha band is 0 outside the rotated
corridor, so fully-transparent blocks are skipped without inference.
Writes a uint8 probability COG (prob*255).
"""
import argparse, importlib.util, os, sys, time
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

SOLAR = "/mnt/drso-data/solar_lora"
MEANS = np.array([123.675, 116.28, 103.53], np.float32)
STDS = np.array([58.395, 57.12, 58.68], np.float32)
TILE, OV = 512, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/drso-data/wayback/bhuj_lakadia_L18_2026-08-05.mbtiles")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()

    sys.path.insert(0, "/home/vivek_jain/seggpt-local/towers_work")
    import custom_modules  # noqa: F401
    from terratorch.tasks import SemanticSegmentationTask
    task = SemanticSegmentationTask.load_from_checkpoint(a.ckpt, map_location="cuda")
    task.eval().to("cuda")
    model = task.model
    print("model:", os.path.basename(a.ckpt), flush=True)

    src = rasterio.open(a.src)
    W, H = src.width, src.height
    print("source: %d x %d  (%.1f Gpx)  bands=%d" % (W, H, W * H / 1e9, src.count), flush=True)

    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8",
                crs=src.crs, transform=src.transform, compress="deflate",
                tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES",
                nodata=0)
    dst = rasterio.open(a.out, "w", **prof)

    step = a.block - OV
    ys = list(range(0, H, step))
    xs = list(range(0, W, step))
    total = len(ys) * len(xs)
    print("blocks: %d x %d = %d" % (len(ys), len(xs), total), flush=True)

    t0 = time.time()
    done = skipped = 0
    px_pos = 0
    for by in ys:
        for bx in xs:
            done += 1
            bh = min(a.block, H - by); bw = min(a.block, W - bx)
            if bh < 64 or bw < 64:
                continue
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
            img = arr[:3]

            prob = np.zeros((bh, bw), np.float32)
            cnt = np.zeros((bh, bw), np.float32)
            ts = TILE - OV
            cys = list(range(0, max(bh - TILE, 0) + 1, ts)) or [0]
            cxs = list(range(0, max(bw - TILE, 0) + 1, ts)) or [0]
            if cys[-1] + TILE < bh: cys.append(bh - TILE)
            if cxs[-1] + TILE < bw: cxs.append(bw - TILE)
            batch, pos = [], []

            def flush():
                if not batch:
                    return
                t = torch.from_numpy(np.stack(batch)).to("cuda")
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
                    if len(batch) >= a.batch:
                        flush()
            flush()

            prob /= np.maximum(cnt, 1)
            prob[~valid] = 0
            out8 = (prob * 255).astype(np.uint8)
            px_pos += int((out8 >= 153).sum())          # >=0.60
            dst.write(out8, 1, window=win)

            if done % 25 == 0 or done == total:
                el = time.time() - t0
                print("[%d/%d] skipped=%d  %.1f min  ETA %.0f min  pos>=0.60: %.2f km2"
                      % (done, total, skipped, el / 60,
                         (total - done) * el / max(done, 1) / 60,
                         px_pos * 0.5971642835 ** 2 / 1e6), flush=True)
    dst.close(); src.close()
    print("DONE in %.1f min | blocks %d (%d empty) | predicted %.2f km2"
          % ((time.time() - t0) / 60, total, skipped, px_pos * 0.5971642835 ** 2 / 1e6))
    print("out:", a.out)


if __name__ == "__main__":
    main()
