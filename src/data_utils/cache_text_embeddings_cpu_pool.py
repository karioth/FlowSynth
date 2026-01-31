import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor, AutoTokenizer, T5EncoderModel

from src.data_utils.audio_utils import scan_cached_outputs

_CLAP_MODEL = None
_CLAP_PROCESSOR = None
_T5_TOK = None
_T5_ENC = None
_WORKER_STATE = {}


def _resolve_node_setting(value, env_keys, default=None):
    if value is not None:
        return value
    for key in env_keys:
        env_val = os.environ.get(key)
        if env_val is None:
            continue
        try:
            return int(env_val)
        except ValueError:
            continue
    return default


def _belongs_to_node(key: str, node_rank: int, node_world_size: int) -> bool:
    if node_world_size <= 1:
        return True
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % int(node_world_size)
    return bucket == int(node_rank)


def _init_worker(
    model_name: str,
    t5_model_name: str,
    device: str,
    output_dir: str,
    worker_threads: int,
):
    global _CLAP_MODEL, _CLAP_PROCESSOR, _T5_TOK, _T5_ENC, _WORKER_STATE

    torch.set_num_threads(int(worker_threads))
    torch.set_num_interop_threads(1)

    _CLAP_MODEL = ClapModel.from_pretrained(model_name, use_safetensors=True).eval().to(device)
    _CLAP_PROCESSOR = ClapProcessor.from_pretrained(model_name)
    _T5_TOK = AutoTokenizer.from_pretrained(t5_model_name)
    _T5_ENC = T5EncoderModel.from_pretrained(t5_model_name, use_safetensors=True).eval().to(device)

    _WORKER_STATE = {
        "device": device,
        "output_dir": Path(output_dir),
    }


@torch.no_grad()
def _encode_clap_text(caption: str):
    inputs = _CLAP_PROCESSOR(
        text=[caption],
        padding=False,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(_WORKER_STATE["device"]) for k, v in inputs.items()}
    text_kwargs = {"input_ids": inputs["input_ids"]}
    if "token_type_ids" in inputs:
        text_kwargs["token_type_ids"] = inputs["token_type_ids"]
    text_out = _CLAP_MODEL.text_model(return_dict=True, output_hidden_states=False, **text_kwargs)

    proj = _CLAP_MODEL.text_projection if hasattr(_CLAP_MODEL, "text_projection") else _CLAP_MODEL.text_model.text_projection
    embeds_unnorm = proj(text_out.pooler_output)
    embeds = F.normalize(embeds_unnorm, dim=-1)
    last_hidden = text_out.last_hidden_state[0]
    return embeds[0], last_hidden, int(last_hidden.shape[0])


@torch.no_grad()
def _encode_t5_tokens(caption: str):
    t5_inputs = _T5_TOK(
        [caption],
        padding=False,
        truncation=True,
        return_tensors="pt",
    )
    t5_inputs = {k: v.to(_WORKER_STATE["device"]) for k, v in t5_inputs.items()}
    t5_out = _T5_ENC(input_ids=t5_inputs["input_ids"], return_dict=True)
    t5_last = t5_out.last_hidden_state[0]
    return t5_last, int(t5_last.shape[0])


@torch.no_grad()
def _process_item(item):
    key, caption = item
    out_path = _WORKER_STATE["output_dir"] / f"{key}.pt"
    if out_path.exists():
        return

    clap_embed, clap_last_hidden, clap_len = _encode_clap_text(caption)
    t5_last_hidden, t5_len = _encode_t5_tokens(caption)

    payload = {
        "clap_embedding": clap_embed.cpu().to(torch.float32),
        "clap_last_hidden": clap_last_hidden.cpu().to(torch.float32),
        "clap_len": int(clap_len),
        "t5_last_hidden": t5_last_hidden.cpu().to(torch.float32),
        "t5_len": int(t5_len),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(out_path) + f".{os.getpid()}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)


def get_args_parser():
    parser = argparse.ArgumentParser("Cache CLAP text embeddings (CPU pool)", add_help=True)
    parser.add_argument("--metadata_path", required=True, type=str, help="Path to manifest.jsonl")
    parser.add_argument("--output_dir", required=True, type=str, help="Output root for text embeddings")
    parser.add_argument("--model_name", default="laion/larger_clap_music", type=str)
    parser.add_argument("--t5_model_name", default="google/flan-t5-large", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--worker_threads", default=1, type=int)
    parser.add_argument("--mp_start_method", default="fork", choices=["fork", "spawn", "forkserver"])
    parser.add_argument("--node_rank", default=None, type=int)
    parser.add_argument("--node_world_size", default=None, type=int)
    return parser


def main(args):
    metadata_path = Path(args.metadata_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    node_rank = _resolve_node_setting(args.node_rank, ["NODE_RANK", "SLURM_NODEID"], default=None)
    node_world_size = _resolve_node_setting(
        args.node_world_size, ["NODE_WORLD_SIZE", "SLURM_JOB_NUM_NODES"], default=1
    )
    if node_world_size > 1 and node_rank is None:
        raise ValueError("node_rank is required when node_world_size > 1")
    if node_rank is None:
        node_rank = 0
    if node_rank < 0 or node_rank >= node_world_size:
        raise ValueError("node_rank must be in [0, node_world_size)")

    if node_rank == 0:
        print(f"Scanning already cached .pt files under: {output_dir}")
    done_cache = scan_cached_outputs(output_dir)
    if node_rank == 0:
        print(f"Found {len(done_cache)} cached files.")

    tasks = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            key = entry["key"]
            caption = entry["caption"]
            rel = f"{key}.pt"
            if rel in done_cache:
                continue
            if not _belongs_to_node(key, node_rank, node_world_size):
                continue
            tasks.append((key, caption))

    print(
        f"Node {node_rank}/{node_world_size}: {len(tasks)} captions to process. "
        f"Cache output: {output_dir}"
    )
    if not tasks:
        return

    ctx = torch.multiprocessing.get_context(args.mp_start_method)
    with ctx.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(
            args.model_name,
            args.t5_model_name,
            args.device,
            str(output_dir),
            args.worker_threads,
        ),
    ) as pool:
        for _ in tqdm(pool.imap_unordered(_process_item, tasks), total=len(tasks), desc="Encoding"):
            pass


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
