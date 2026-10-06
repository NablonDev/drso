#!/usr/bin/env python
"""Install LULC model weights into <bucket>/models with a sha256 manifest.

Copies (never moves) the checkpoints and the Hugging Face cache of the gated DINOv3 base from
their research locations, so run time never touches those folders. On the production VM, run
it with --from <dir> pointing at a copy of <bucket>/models taken from the dev box.

    python scripts/install_models.py            # dev: copy from the research paths below
    python scripts/install_models.py --verify   # check sizes + sha256 against the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from beeline_api import settings  # noqa: E402

SOURCES = {
    "buildings/best.pth": "/home/vivek_jain/bhuj-lakadia/output/finetune_dataset/checkpoints/best.pth",
    "village_roads/HF_GG08_60.pkl": "/mnt/cog-data/roaddet/HF_GG08_60.pkl",
    "solar/epoch07.ckpt": "/mnt/drso-data/solar_lora/checkpoints/dinov3_lora_solar_v1/lightning_logs/version_1/checkpoints/epoch07.ckpt",
    "solar/hparams.yaml": "/mnt/drso-data/solar_lora/checkpoints/dinov3_lora_solar_v1/lightning_logs/version_1/hparams.yaml",
    "substations/stageB_epoch22.ckpt": "/mnt/cog-data/vivek/stageA/checkpoints/stageB/lightning_logs/version_0/checkpoints/epoch22.ckpt",
    "substations/hparams.yaml": "/mnt/cog-data/vivek/stageA/checkpoints/stageB/lightning_logs/version_0/hparams.yaml",
    "wind_turbines/round4_epoch25.ckpt": "/home/vivek_jain/seggpt-local/turbines_work/dinov3_lora_turbine_round4/lightning_logs/version_0/checkpoints/epoch=25.ckpt",
    "wind_turbines/hparams.yaml": "/home/vivek_jain/seggpt-local/turbines_work/dinov3_lora_turbine_round4/lightning_logs/version_0/hparams.yaml",
}
HF_SRC = Path.home() / ".cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m"
HF_DST = "hf_home/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--from", dest="src", help="directory holding a previously installed models/ tree")
    a = ap.parse_args()
    root = settings.DATA / "models"
    man_p = root / "MANIFEST.json"
    if a.verify:
        man = json.loads(man_p.read_text())
        bad = [k for k, v in man["files"].items()
               if not (root / k).exists() or (root / k).stat().st_size != v["size"] or sha256(root / k) != v["sha256"]]
        print("OK" if not bad else f"MISMATCH: {bad}")
        sys.exit(1 if bad else 0)
    root.mkdir(parents=True, exist_ok=True)
    pairs = []
    if a.src:
        s = Path(a.src)
        pairs = [(p, root / p.relative_to(s)) for p in s.rglob("*") if p.is_file()]
    else:
        pairs = [(Path(src), root / dst) for dst, src in SOURCES.items()]
        for p in HF_SRC.rglob("*"):
            if "blobs" in p.relative_to(HF_SRC).parts:
                continue          # snapshots are copied by value; the blob store would double the size
            if p.is_file() or p.is_symlink():
                pairs.append((p.resolve(), root / HF_DST / p.relative_to(HF_SRC)))
    files = {}
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        rel = str(dst.relative_to(root))
        if rel == "MANIFEST.json":
            continue
        files[rel] = {"size": dst.stat().st_size, "sha256": sha256(dst), "from": str(src)}
        print(f"{rel}  {dst.stat().st_size / 1e6:.0f} MB")
    man_p.write_text(json.dumps({"installed_at": datetime.now(timezone.utc).isoformat(), "files": files}, indent=1))
    print(f"manifest: {man_p}  ({sum(v['size'] for v in files.values()) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
