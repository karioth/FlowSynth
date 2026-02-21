#!/usr/bin/env python3
"""
Run AudioLDM-style objective eval (audioldm_eval) on two folders of WAVs.
Usage:
  python evaluate.py --gen /path/to/gen_wavs --gt /path/to/gt_wavs
"""

import argparse
import json
from pathlib import Path
import urllib.request

import torch
from audioldm_eval import EvaluationHelper


def list_wavs(d: Path) -> set[str]:
    return {p.name for p in d.glob("*.wav")}


CNN14_CKPT_URLS = {
    "Cnn14_mAP=0.431.pth": "https://zenodo.org/record/3576403/files/Cnn14_mAP%3D0.431.pth",
    "Cnn14_16k_mAP=0.438.pth": "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
}


def ensure_cnn14_checkpoints() -> None:
    ckpt_dir = Path.home() / ".cache" / "audioldm_eval" / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in CNN14_CKPT_URLS.items():
        target = ckpt_dir / filename
        if target.exists():
            continue
        tmp_target = target.with_suffix(target.suffix + ".tmp")
        print(f"[info] downloading {filename} ...")
        urllib.request.urlretrieve(url, tmp_target)
        tmp_target.replace(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, required=True, help="Folder with generated .wav files")
    ap.add_argument("--gt", type=Path, required=True, help="Folder with ground-truth .wav files")
    ap.add_argument("--sr", type=int, default=16000, help="Eval sample rate (use 16000 for AudioCaps protocol)")
    ap.add_argument("--backbone", type=str, default="cnn14", choices=["cnn14", "mert"])
    ap.add_argument("--limit", type=int, default=0, help="If >0, evaluate only this many files (smoke test)")
    args = ap.parse_args()

    if args.backbone == "cnn14":
        ensure_cnn14_checkpoints()

    gen_dir = args.gen.resolve()
    gt_dir = args.gt.resolve()
    if not gen_dir.is_dir() or not gt_dir.is_dir():
        raise SystemExit(f"gen_dir or gt_dir does not exist: {gen_dir} {gt_dir}")

    gen_names = list_wavs(gen_dir)
    gt_names = list_wavs(gt_dir)
    inter = gen_names & gt_names

    if len(inter) == 0:
        raise SystemExit("No overlapping .wav basenames between --gen and --gt (paired mode requires matching names).")

    if gen_names != gt_names:
        print(f"[warn] gen wavs: {len(gen_names)}, gt wavs: {len(gt_names)}, intersection: {len(inter)}")
        missing_in_gen = sorted(gt_names - gen_names)[:10]
        missing_in_gt = sorted(gen_names - gt_names)[:10]
        if missing_in_gen:
            print("[warn] examples missing in --gen:", missing_in_gen)
        if missing_in_gt:
            print("[warn] examples missing in --gt:", missing_in_gt)

    limit_num = args.limit if args.limit > 0 else None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluator = EvaluationHelper(args.sr, device, backbone=args.backbone)
    metrics = evaluator.main(str(gen_dir), str(gt_dir), limit_num=limit_num)

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
