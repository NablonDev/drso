"""Per-block 4-scale union road probability (from guj/code/guj_infer.py).

Same model, scales, halo and output encoding as the statewide run: the block is inferred on a
(NT+2H)^2 tile mosaic, the NT*256 px core is cropped and written as two Byte COGs (0-255),
u_<bx>_<by>.tif (max over scales) and n_<bx>_<by>.tif (native 0.6 m scale, used for width).
TensorFlow is imported only by load_model(), so this module is safe to import without a GPU.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
from PIL import Image

from aoi_grid import H, NT, block_transform


def load_model(checkpoint: str, vram_mb: int | None = None):
    if vram_mb:
        os.environ["ROADDET_VRAM_MB"] = str(int(vram_mb))      # read by infer.py at import
    import infer
    model = infer.get_res_unet6(infer.IN, infer.IN)
    infer.load_weights(model, checkpoint)
    return model


def predict_block(model, mos, scales=(1.0, 1.5, 2.01, 3.0), batch_size=4, halo=H):
    """mos: (T*256, T*256, 3) uint8. Returns (union_core, native_core) float32 (NT*256)^2."""
    import infer
    px = mos.shape[0]; c0 = halo * 256; core = NT * 256
    rgb0 = np.transpose(mos, (2, 0, 1)).astype(np.float32) / 255.0
    acc = None; native = None
    for sc in scales:
        if sc == 1.0:
            rgb = rgb0
        else:
            h = w = int(px / sc)
            rgb = np.stack([np.asarray(Image.fromarray(mos[..., b]).resize((w, h), Image.BOX),
                                       dtype=np.float32) / 255.0 for b in range(3)])
        p = infer.predict_array(model, rgb, batch_size=batch_size)
        if sc != 1.0:
            p = np.asarray(Image.fromarray(p).resize((px, px), Image.BILINEAR))
        else:
            native = p[c0:c0 + core, c0:c0 + core].copy()
        acc = p if acc is None else np.maximum(acc, p)
    if native is None:                       # 1.0 not among the scales: width from the union
        native = acc[c0:c0 + core, c0:c0 + core].copy()
    return acc[c0:c0 + core, c0:c0 + core], native


def write_cog(arr, tr, dst):
    """float32 [0,1] -> Byte COG (0-255), atomically (tmp + rename)."""
    import rasterio
    tmp = dst[:-4] + "._t.tif"; part = dst[:-4] + "._p.tif"
    with rasterio.open(tmp, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs="EPSG:3857", transform=tr, compress="deflate",
                       predictor=2, tiled=True) as d:
        d.write(arr.astype(np.float32), 1)
    try:
        subprocess.run(["gdal_translate", "-q", "-of", "COG", "-ot", "Byte", "-scale", "0", "1", "0", "255",
                        "-a_nodata", "none", "-co", "COMPRESS=DEFLATE", "-co", "BLOCKSIZE=512",
                        "-co", "OVERVIEW_RESAMPLING=NEAREST", tmp, part], check=True)
        os.replace(part, dst)
    finally:
        for f in (tmp, part):
            if os.path.exists(f):
                os.remove(f)


def write_block(bx, by, union, native, u_dir, n_dir):
    tr = block_transform(bx, by)
    du = os.path.join(u_dir, "u_%d_%d.tif" % (bx, by))
    dn = os.path.join(n_dir, "n_%d_%d.tif" % (bx, by))
    write_cog(native, tr, dn)        # n_ first: u_ existing marks the block as finished
    write_cog(union, tr, du)
    return du, dn
