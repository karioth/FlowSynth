import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor


class CaptionDataset(Dataset):
    """
    Loads captions from JSONL and returns (embedding_id, caption_text).
    embedding_id format: {track_id}_{start:03d}-{end:03d}
    """
    def __init__(self, metadata_path: str):
        self.metadata_path = Path(metadata_path)
        self.entries = []

        with open(self.metadata_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    self.entries.append(entry)
                except json.JSONDecodeError:
                    continue

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        track_id = entry['id']
        start = int(entry['start_time'])
        end = int(entry['end_time'])
        caption = entry['caption']

        embedding_id = f"{track_id}_{start:03d}-{end:03d}"
        return embedding_id, caption


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
                        help="Path to final_caption30sec.jsonl")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Output directory for text embeddings")
    parser.add_argument("--model_name", default="laion/larger_clap_music", type=str,
                        help="HuggingFace model ID")
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
        embedding_ids = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        # Check which embeddings already exist
        to_process_ids = []
        to_process_captions = []
        for emb_id, caption in zip(embedding_ids, captions):
            out_path = output_dir / f"{emb_id}.pt"
            if not out_path.exists():
                to_process_ids.append(emb_id)
                to_process_captions.append(caption)

        if not to_process_ids:
            continue

        # Process text through CLAP
        inputs = processor(
            text=to_process_captions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        text_embeds = model.get_text_features(**inputs)

        # Save embeddings
        for emb_id, text_embed in zip(to_process_ids, text_embeds):
            out_path = output_dir / f"{emb_id}.pt"
            payload = {
                "text_embedding": text_embed.cpu().to(torch.float32),
            }

            tmp_path = str(out_path) + ".tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, out_path)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
