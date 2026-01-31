import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor, AutoTokenizer, T5EncoderModel


class CaptionDataset(Dataset):
    """
    Loads captions from JSONL and returns (key, caption_text).
    Manifest JSONL entries contain:
      - key: str
      - caption: str
    """
    def __init__(self, metadata_path: str):
        self.metadata_path = Path(metadata_path)
        self.entries = []

        with open(self.metadata_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        key = entry["key"]
        caption = entry["caption"]
        return key, caption


def init_distributed_mode(device: str):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        backend = "nccl" if torch.cuda.is_available() and device.startswith("cuda") else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        return True, rank, world_size
    return False, 0, 1


def get_args_parser():
    parser = argparse.ArgumentParser("Cache CLAP text embeddings", add_help=True)
    parser.add_argument("--metadata_path", required=True, type=str,
                        help="Path to manifest.jsonl")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Output root for text embeddings")
    parser.add_argument("--model_name", default="laion/larger_clap_music", type=str,
                        help="HuggingFace model ID")
    parser.add_argument("--t5_model_name", default="google/flan-t5-large", type=str,
                        help="HuggingFace model ID for Flan-T5 encoder")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size per GPU")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use for encoding")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of DataLoader workers")
    parser.add_argument("--pin_mem", action="store_true",
                        help="Pin CPU memory in DataLoader for efficient GPU transfer")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    return parser


def _out_path(output_dir: Path, key: str) -> Path:
    p = Path(key)
    if p.suffix != ".pt":
        p = p.with_suffix(".pt")
    return output_dir / p


@torch.no_grad()
def encode_clap_text(model: ClapModel, processor: ClapProcessor, captions: list[str], device: str):
    # Tokenize once; we’ll slice away padding when saving.
    inputs = processor(
        text=captions,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    text_kwargs = {k: inputs[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in inputs}
    text_out = model.text_model(return_dict=True, output_hidden_states=False, **text_kwargs)

    proj = model.text_projection if hasattr(model, "text_projection") else model.text_model.text_projection
    embeds_unnorm = proj(text_out.pooler_output)          # [B, 512]
    embeds = F.normalize(embeds_unnorm, dim=-1)           # matches get_text_features in your setup
    lengths = inputs["attention_mask"].sum(dim=-1).to(torch.int64)  # [B], excludes padding
    return embeds, text_out.last_hidden_state, lengths


@torch.no_grad()
def encode_t5_tokens(t5_tok, t5_enc, captions: list[str], device: str):
    t5_inputs = t5_tok(
        captions,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    t5_inputs = {k: v.to(device) for k, v in t5_inputs.items()}
    t5_out = t5_enc(**t5_inputs, return_dict=True)
    t5_last = t5_out.last_hidden_state  # [B, Lt, 1024]
    t5_lens = t5_inputs["attention_mask"].sum(dim=-1).to(torch.int64)
    return t5_last, t5_lens


@torch.no_grad()
def main(args):
    device = args.device
    distributed, rank, world_size = init_distributed_mode(device)

    output_dir = Path(args.output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    if rank == 0:
        print(f"Loading CLAP model: {args.model_name}")

    model = ClapModel.from_pretrained(args.model_name, use_safetensors=True).eval().to(device)
    processor = ClapProcessor.from_pretrained(args.model_name)

    if rank == 0:
        print(f"Loading Flan-T5 encoder: {args.t5_model_name}")
    t5_tok = AutoTokenizer.from_pretrained(args.t5_model_name)
    t5_enc = T5EncoderModel.from_pretrained(args.t5_model_name, use_safetensors=True).eval().to(device)

    if rank == 0:
        print(f"Loading captions from: {args.metadata_path}")

    dataset = CaptionDataset(args.metadata_path)

    if rank == 0:
        print(f"Total captions: {len(dataset):,}")

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
        collate_fn=lambda b: b,
    )

    use_tqdm = rank == 0
    data_iter = loader
    if use_tqdm:
        data_iter = tqdm(loader, total=len(loader), desc="Encoding", unit="batch")

    for batch in data_iter:
        keys = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        # Check which embeddings already exist
        to_process_ids = []
        to_process_captions = []
        to_process_paths = []
        for key, caption in zip(keys, captions):
            out_path = _out_path(output_dir, key)
            if not out_path.exists():
                to_process_ids.append(key)
                to_process_captions.append(caption)
                to_process_paths.append(out_path)

        if not to_process_ids:
            continue

        clap_embeds, clap_last_hidden, clap_lens = encode_clap_text(model, processor, to_process_captions, device)
        t5_last_hidden, t5_lens = encode_t5_tokens(t5_tok, t5_enc, to_process_captions, device)

        # Save embeddings
        for i, out_path in enumerate(to_process_paths):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Lc = int(clap_lens[i].item())
            Lt = int(t5_lens[i].item())
            payload = {
                "clap_embedding": clap_embeds[i].cpu().to(torch.float32),                    # [512], normalized
                "clap_last_hidden": clap_last_hidden[i, :Lc].cpu().to(torch.float32),        # [Lc, 768], no padding
                "clap_len": Lc,
                "flan_t5_last_hidden": t5_last_hidden[i, :Lt].cpu().to(torch.float32),       # [Lt, 1024], no padding
                "flan_t5_len": Lt,
            }

            tmp_path = str(out_path) + ".tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, out_path)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
