#!/usr/bin/env python3
"""
Run AudioLDM-style objective eval (audioldm_eval) on two folders of WAVs.
Usage:
  python evaluate.py --gen /path/to/gen_wavs --gt /path/to/gt_wavs
"""

import argparse
from contextlib import contextmanager
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

MERT_MODEL_ID = "m-a-p/MERT-v1-95M"


def torch_is_below_2_6() -> bool:
    # Avoid extra dependencies for version parsing.
    version = torch.__version__.split("+", 1)[0]
    nums = []
    for part in version.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        nums.append(int(digits))
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3]) < (2, 6, 0)


def ensure_mert_safetensors_snapshot(model_id: str = MERT_MODEL_ID) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(snapshot_download(repo_id=model_id))
    safe_file = snapshot_dir / "model.safetensors"
    if safe_file.exists():
        return snapshot_dir

    bin_file = snapshot_dir / "pytorch_model.bin"
    if not bin_file.exists():
        raise RuntimeError(f"Missing checkpoint file: {bin_file}")

    from safetensors.torch import save_file

    print(f"[info] converting {bin_file.name} -> {safe_file.name} (one-time setup)")
    state_dict = torch.load(bin_file, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise RuntimeError("Unexpected checkpoint format while converting MERT weights.")

    tensor_state = {k: v.detach().cpu() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    if not tensor_state:
        raise RuntimeError("No tensors found in MERT checkpoint during conversion.")

    try:
        save_file(tensor_state, str(safe_file))
    except RuntimeError as exc:
        # Some checkpoints have tied/shared storage; cloned tensors avoid save errors.
        if "share" not in str(exc).lower():
            raise
        print("[info] retrying safetensors export with cloned tensors")
        save_file({k: v.clone() for k, v in tensor_state.items()}, str(safe_file))

    return snapshot_dir


@contextmanager
def patch_mert_loader_for_safetensors(snapshot_dir: Path | None):
    if snapshot_dir is None:
        yield
        return

    import audioldm_eval.eval as eval_impl

    original_auto_from_pretrained = eval_impl.AutoModel.from_pretrained

    def patched_auto_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        if str(pretrained_model_name_or_path).rstrip("/") == MERT_MODEL_ID:
            kwargs = dict(kwargs)
            kwargs["use_safetensors"] = True
            return original_auto_from_pretrained(str(snapshot_dir), *args, **kwargs)
        return original_auto_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    eval_impl.AutoModel.from_pretrained = patched_auto_from_pretrained
    try:
        yield
    finally:
        eval_impl.AutoModel.from_pretrained = original_auto_from_pretrained


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
    mert_snapshot = None
    if args.backbone == "mert" and torch_is_below_2_6():
        mert_snapshot = ensure_mert_safetensors_snapshot()

    with patch_mert_loader_for_safetensors(mert_snapshot):
        evaluator = EvaluationHelper(args.sr, device, backbone=args.backbone)
    metrics = evaluator.main(str(gen_dir), str(gt_dir), limit_num=limit_num)

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
