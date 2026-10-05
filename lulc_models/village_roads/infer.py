# Copied from roaddet/infer.py (2026-09-23). Unchanged except this header: the VRAM cap is read
# from ROADDET_VRAM_MB at import, so run_aoi.py sets it (from request gpu.vram_mb) before importing.
"""Run the Microsoft RoadDetections res-unet-6 over a georeferenced chip.

The published weights are a pickle of Keras-2 `model.get_weights()` (channels-first,
Theano-style dim ordering).  We rebuild the identical graph with tf_keras (the
Keras 2 API on TF2) so the weight ordering matches exactly, then verify shapes.
"""
import os, sys, pickle, argparse, math
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

_cap = os.environ.get("ROADDET_VRAM_MB")
for g in tf.config.list_physical_devices("GPU"):
    if _cap:   # hard ceiling so a restarted llama-server still fits beside us
        tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(memory_limit=int(_cap))])
    else:
        tf.config.experimental.set_memory_growth(g, True)   # don't hog the A100

import tf_keras as keras
from tf_keras import backend as K
from tf_keras.layers import (Input, Conv2D, UpSampling2D, Add, Concatenate,
                             Activation, BatchNormalization)
from tf_keras.models import Model

K.set_image_data_format("channels_first")

import rasterio
from rasterio.enums import Resampling

# ---------------------------------------------------------------- architecture
# Verbatim port of RoadDetections/model/code/unet.py :: get_res_unet6
NUM_CH, NUM_MASK_CH = 3, 1


def bn_act(x, act=True):
    x = BatchNormalization(axis=1)(x)
    return Activation("relu")(x) if act else x


def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = bn_act(x)
    return Conv2D(filters, kernel_size, padding=padding,
                  kernel_initializer="he_uniform", strides=strides)(c)


def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding,
                  kernel_initializer="he_uniform", strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size, padding, strides)
    shortcut = Conv2D(filters, (1, 1), padding=padding,
                      kernel_initializer="he_uniform", strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    return Add()([conv, shortcut])


def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size, padding, strides)
    res = conv_block(res, filters, kernel_size, padding, 1)
    shortcut = Conv2D(filters, (1, 1), padding=padding,
                      kernel_initializer="he_uniform", strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    return Add()([shortcut, res])


def upsample_concat_block(x, xskip):
    return Concatenate(axis=1)([UpSampling2D((2, 2))(x), xskip])


def get_res_unet6(rows, cols):
    f = [32, 64, 128, 256, 512, 1024, 2048]
    inputs = Input((NUM_CH, rows, cols))
    e1 = stem(inputs, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    e6 = residual_block(e5, f[5], strides=2)
    b1 = conv_block(e6, f[6], strides=2)
    b2 = conv_block(b1, f[6], strides=1)
    up1 = upsample_concat_block(b2, e6);  d1 = residual_block(up1, f[5])
    up2 = upsample_concat_block(d1, e5);  d2 = residual_block(up2, f[4])
    up3 = upsample_concat_block(d2, e4);  d3 = residual_block(up3, f[3])
    up4 = upsample_concat_block(d3, e3);  d4 = residual_block(up4, f[2])
    up5 = upsample_concat_block(d4, e2);  d5 = residual_block(up5, f[1])
    up6 = upsample_concat_block(d5, e1);  d6 = residual_block(up6, f[0])
    outputs = Conv2D(NUM_MASK_CH, 1, activation="sigmoid")(d6)
    return Model(inputs=inputs, outputs=outputs)


# ------------------------------------------------------------------- weights
class _Numpy1Unpickler(pickle.Unpickler):
    """numpy>=2 moved numpy.core -> numpy._core; remap for this 2018-era pickle."""
    def find_class(self, module, name):
        if module.startswith("numpy.core"):
            module = module.replace("numpy.core", "numpy._core", 1)
        return super().find_class(module, name)


def load_weights(model, path):
    with open(path, "rb") as f:
        try:
            w = pickle.load(f)
        except (ModuleNotFoundError, AttributeError):
            f.seek(0)
            w = _Numpy1Unpickler(f).load()

    want = [tuple(v.shape) for v in model.weights]
    have = [tuple(np.shape(a)) for a in w]
    if len(want) != len(have):
        raise SystemExit(f"weight COUNT mismatch: model {len(want)} vs pickle {len(have)}")
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(want, have)) if a != b]
    if bad:
        for i, a, b in bad[:10]:
            print(f"  [{i}] model {a} != pickle {b}")
        raise SystemExit(f"weight SHAPE mismatch at {len(bad)}/{len(want)} tensors")
    model.set_weights(w)
    print(f"weights OK: {len(want)} tensors, all shapes match "
          f"({sum(np.prod(s) for s in want)/1e6:.1f}M params)")


# ----------------------------------------------------------------- inference
IN = 1088
PAD = 32               # repo cuts 32 px off each side of the prediction
CORE = IN - 2 * PAD    # 1024


def predict_array(model, rgb, batch_size=4):
    """rgb: (3,H,W) float32 in [0,1]. Returns (H,W) float32 probabilities."""
    _, H, W = rgb.shape
    nY = math.ceil(H / CORE)
    nX = math.ceil(W / CORE)
    padded = np.pad(rgb, ((0, 0), (PAD, nY * CORE - H + PAD), (PAD, nX * CORE - W + PAD)),
                    mode="reflect")
    out = np.zeros((H, W), dtype=np.float32)

    coords, batch = [], []
    def flush():
        if not batch:
            return
        pred = model.predict(np.stack(batch), verbose=0)[:, 0]     # (B,1088,1088)
        for (yy, xx), p in zip(coords, pred):
            core = p[PAD:PAD + CORE, PAD:PAD + CORE]
            h = min(CORE, H - yy); w = min(CORE, W - xx)
            out[yy:yy + h, xx:xx + w] = core[:h, :w]
        coords.clear(); batch.clear()

    for iy in range(nY):
        for ix in range(nX):
            yy, xx = iy * CORE, ix * CORE
            batch.append(padded[:, yy:yy + IN, xx:xx + IN].astype(np.float32))
            coords.append((yy, xx))
            if len(batch) == batch_size:
                flush()
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--downscale", type=int, default=1,
                    help="2 = resample to ~1.1 m/px to match the model's training GSD")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    with rasterio.open(args.src) as src:
        prof = src.profile
        H, W = src.height, src.width
        if args.downscale > 1:
            h, w = H // args.downscale, W // args.downscale
            rgb = src.read(out_shape=(3, h, w), resampling=Resampling.average)
        else:
            h, w = H, W
            rgb = src.read()
    rgb = rgb.astype(np.float32) / 255.0
    print(f"input {args.src}: {W}x{H} -> inference grid {w}x{h} (downscale {args.downscale})")

    model = get_res_unet6(IN, IN)
    load_weights(model, args.weights)

    prob = predict_array(model, rgb, args.batch_size)

    if args.downscale > 1:   # put probabilities back on the native pixel grid
        import PIL.Image as PILImage
        prob = np.asarray(PILImage.fromarray(prob).resize((W, H), PILImage.BILINEAR))

    prof.update(count=1, dtype="float32", compress="deflate", predictor=2)
    with rasterio.open(args.out, "w", **prof) as dst:
        dst.write(prob.astype(np.float32), 1)
    print(f"wrote {args.out}  prob: min {prob.min():.3f} max {prob.max():.3f} "
          f"mean {prob.mean():.4f}  frac>0.5 {(prob>0.5).mean()*100:.2f}%")


if __name__ == "__main__":
    main()
