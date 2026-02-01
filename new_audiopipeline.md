# Audio Pipeline Update Plan

## Summary
Update LatentLM audio training pipeline to support:
1. New dataset structure (WavCaps/AudioCaps with separate text_embeddings and audio_latents folders)
2. 69-token prompt conditioning (pooled CLAP + T5 hidden states)
3. Variable-length audio with silence padding/random cropping

---

## Files to Modify

### 1. `src/data_utils/audio_datamodule.py`
Create new `MultiSourceAudioDataset` class:
- Load from multiple dataset sources (WavCaps subsets + AudioCaps train)
- Match audio_latents/*.pt with text_embeddings/*.pt by filename
- Audio handling:
  - Shorter than 251: pad with silence from `silence_samples/silence_10s_dacvae.pt`
  - Longer than 251: random crop
- Text handling:
  - Return dict with `clap` [512], `t5` [68, 1024], `t5_len`
  - Pad T5 to 68 tokens if shorter, truncate if longer
- Custom collate function to batch the dict structure

### 2. `src/models/modules/embeddings.py`
Create new `SequencePromptEmbedder` class:
- Two projection layers: `clap_proj` (512 → hidden_size), `t5_proj` (1024 → hidden_size)
- Learnable `null_embeddings` [69, hidden_size] for padding and CFG dropout
- Forward:
  - Project CLAP → [B, 1, hidden_size]
  - Project T5 → [B, 68, hidden_size], replace padded positions with null
  - Concat → [B, 69, hidden_size]
  - CFG dropout: replace ALL 69 tokens with null_embeddings

### 3. `src/models/AR_DiT.py`
- Replace `ContinuousEmbedder` with `SequencePromptEmbedder`
- Forward: prepend 69 prompt tokens (not 1), zero time modulation for prompt tokens
- Output: remove first 69 tokens before final layer
- Update `sample_with_cfg` for dict prompt structure

### 4. `src/models/DiT.py`
- Replace `ContinuousEmbedder` with `SequencePromptEmbedder`
- Forward: prepend 69 tokens, remove after blocks
- Update `sample_with_cfg` for dict prompt structure

### 5. `src/models/Transformer.py`
- Replace `ContinuousEmbedder` with `SequencePromptEmbedder`
- `forward_parallel`: prepend 69 tokens, shift input by 69 instead of 1
- `forward_recurrent`: process 69 prompt tokens at start_pos=0

### 6. `src/models/MaskedAR.py`
- Replace `ContinuousEmbedder` with `SequencePromptEmbedder`
- `forward_backbone`: prepend 69 tokens, remove after blocks
- `forward_recurrent`: handle 69 prompt tokens at position 0

### 7. `src/lightning.py`
- Add params: `clap_dim`, `t5_dim`, `prompt_seq_len`
- Add `conditioning_type="sequence"` handling
- `training_step`: unpack `prompt_data` dict

### 8. `train_audio.py`
New arguments:
- `--wavcaps-root` (default: /share/users/student/f/friverossego/datasets/WavCaps)
- `--audiocaps-root` (default: /share/users/student/f/friverossego/datasets/AudioCaps)
- `--silence-latent-path` (default: silence_samples/silence_10s_dacvae.pt)
- `--clap-dim` (512), `--t5-dim` (1024), `--prompt-seq-len` (69)

Build dataset configs for WavCaps subsets (AudioSet_SL, BBC_Sound_Effects, FreeSound, SoundBible) + AudioCaps train

### 9. `sample_audio.py`
- Add T5 model loading (`google/flan-t5-large`)
- New `get_text_embeddings()` function returning dict with clap/t5/t5_len
- Pass dict to `sample_latents()` instead of single tensor

---

## Implementation Order

1. **Data Pipeline** - `audio_datamodule.py`
   - Test loading independently before model changes

2. **Embedding Layer** - `embeddings.py`
   - Add `SequencePromptEmbedder`, keep old `ContinuousEmbedder`

3. **Models** (one at a time)
   - MaskedAR.py (primary model for audio)
   - AR_DiT.py
   - DiT.py
   - Transformer.py

4. **Training Script** - `train_audio.py`, `lightning.py`

5. **Inference** - `sample_audio.py`

---

## Key Dimensions
| Component | Shape |
|-----------|-------|
| CLAP pooled | [B, 512] |
| T5 hidden | [B, 68, 1024] |
| Prompt sequence | [B, 69, hidden_size] |
| Audio latent | [B, 251, 256] |
| Silence latent | [256, ~250] (tile as needed) |

---

## Verification
1. Run data loading test: load batch, verify shapes match expected
2. Run single training step with gradient check
3. Full training run on small subset
4. Inference test with sample_audio.py

---

# Step 1: Data Pipeline - Detailed Plan

## File: `src/data_utils/audio_datamodule.py`

### Current State
```python
class CachedAudioDataset(Dataset):
    # Loads single .pt files with combined audio + text
    # Fixed 251 sequence length, raises error if mismatch
    # Returns: (moments [251, 256], clap_embedding [512])
```

### New Classes to Add

#### 1. `MultiSourceAudioDataset`

```python
class MultiSourceAudioDataset(Dataset):
    """
    Loads audio latents and text embeddings from separate directories.
    Supports multiple dataset sources (WavCaps subsets + AudioCaps).
    """

    def __init__(
        self,
        dataset_roots: list[str],      # List of dataset root paths
        silence_latent_path: str,       # Path to silence_10s_dacvae.pt
        target_seq_len: int = 251,      # Target audio sequence length
        max_t5_tokens: int = 68,        # Max T5 tokens (prompt_seq_len - 1)
    ):
        ...
```

**Constructor logic:**
1. Load silence latent from `silence_latent_path`
   - Extract `posterior_params` → transpose to [T, 2C]
   - Store as `self.silence_latent`
2. For each root in `dataset_roots`:
   - Scan `{root}/audio_latents/*.pt` for all files
   - For each audio file, verify matching `{root}/text_embeddings/{same_name}.pt` exists
   - Store tuples: `(audio_path, text_path)`
3. Concatenate all file tuples into `self.files`

**`__getitem__` logic:**
```python
def __getitem__(self, idx):
    audio_path, text_path = self.files[idx]

    # 1. Load audio latent
    audio_data = torch.load(audio_path, map_location="cpu", weights_only=True)
    posterior_params = audio_data["posterior_params"]  # [256, T_var]
    latent_length = audio_data["latent_length"]
    moments = posterior_params.transpose(0, 1)  # [T_var, 256]

    # 2. Adjust audio length
    moments = self._adjust_audio_length(moments)  # [251, 256]

    # 3. Load text embeddings
    text_data = torch.load(text_path, map_location="cpu", weights_only=True)
    clap_emb = text_data["clap_embedding"]       # [512]
    t5_hidden = text_data["t5_last_hidden"]      # [T5_len, 1024]
    t5_len = text_data["t5_len"]                 # int

    # 4. Prepare T5 (pad/truncate to max_t5_tokens)
    t5_padded = self._prepare_t5(t5_hidden, t5_len)  # [68, 1024]

    # 5. Return structured prompt data
    prompt_data = {
        "clap": clap_emb,           # [512]
        "t5": t5_padded,            # [68, 1024]
        "t5_len": min(t5_len, self.max_t5_tokens),  # int
    }

    return moments, prompt_data
```

**Helper methods:**

```python
def _adjust_audio_length(self, moments: torch.Tensor) -> torch.Tensor:
    """Pad with silence or random crop to target_seq_len."""
    current_len = moments.shape[0]

    if current_len == self.target_seq_len:
        return moments

    if current_len < self.target_seq_len:
        # Pad with silence
        pad_needed = self.target_seq_len - current_len
        silence_len = self.silence_latent.shape[0]

        # Tile silence if needed, then take what we need
        num_tiles = (pad_needed // silence_len) + 1
        silence_tiled = self.silence_latent.repeat(num_tiles, 1)
        padding = silence_tiled[:pad_needed]

        return torch.cat([moments, padding], dim=0)

    else:  # current_len > target_seq_len
        # Random crop
        max_start = current_len - self.target_seq_len
        start = torch.randint(0, max_start + 1, (1,)).item()
        return moments[start : start + self.target_seq_len]


def _prepare_t5(self, t5_hidden: torch.Tensor, t5_len: int) -> torch.Tensor:
    """Pad or truncate T5 to max_t5_tokens."""
    if t5_len > self.max_t5_tokens:
        # Truncate
        return t5_hidden[:self.max_t5_tokens].clone()

    if t5_len < self.max_t5_tokens:
        # Pad with zeros
        pad_size = self.max_t5_tokens - t5_len
        padding = torch.zeros(pad_size, t5_hidden.shape[1], dtype=t5_hidden.dtype)
        return torch.cat([t5_hidden, padding], dim=0)

    return t5_hidden.clone()
```

#### 2. Custom Collate Function

```python
def audio_collate_fn(batch: list) -> tuple[torch.Tensor, dict]:
    """
    Collate batch of (moments, prompt_data) into batched tensors.

    Returns:
        moments: [B, 251, 256]
        prompt_data: {
            "clap": [B, 512],
            "t5": [B, 68, 1024],
            "t5_len": [B] (LongTensor)
        }
    """
    moments_list = [item[0] for item in batch]
    prompt_list = [item[1] for item in batch]

    moments = torch.stack(moments_list, dim=0)

    prompt_data = {
        "clap": torch.stack([p["clap"] for p in prompt_list], dim=0),
        "t5": torch.stack([p["t5"] for p in prompt_list], dim=0),
        "t5_len": torch.tensor([p["t5_len"] for p in prompt_list], dtype=torch.long),
    }

    return moments, prompt_data
```

#### 3. Updated `CachedAudioDataModule`

```python
class CachedAudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_roots: list[str],       # NEW: list of roots instead of single path
        silence_latent_path: str,       # NEW
        batch_size: int,
        target_seq_len: int = 251,      # renamed from seq_len
        max_t5_tokens: int = 68,        # NEW
        num_workers: int = 4,
    ):
        super().__init__()
        self.dataset_roots = dataset_roots
        self.silence_latent_path = silence_latent_path
        self.batch_size = batch_size
        self.target_seq_len = target_seq_len
        self.max_t5_tokens = max_t5_tokens
        self.num_workers = num_workers
        self.train_dataset = None

    def setup(self, stage=None):
        self.train_dataset = MultiSourceAudioDataset(
            dataset_roots=self.dataset_roots,
            silence_latent_path=self.silence_latent_path,
            target_seq_len=self.target_seq_len,
            max_t5_tokens=self.max_t5_tokens,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            collate_fn=audio_collate_fn,  # NEW
        )
```

### Dataset Roots Configuration

When called from `train_audio.py`, build roots list:
```python
dataset_roots = [
    "/share/users/student/f/friverossego/datasets/WavCaps/AudioSet_SL",
    "/share/users/student/f/friverossego/datasets/WavCaps/BBC_Sound_Effects",
    "/share/users/student/f/friverossego/datasets/WavCaps/FreeSound",
    "/share/users/student/f/friverossego/datasets/WavCaps/SoundBible",
    "/share/users/student/f/friverossego/datasets/AudioCaps/train",
]
```

### Data File Formats Reference

**Audio latent file** (`audio_latents/{id}.pt`):
```python
{
    "posterior_params": torch.Tensor,  # [256, T_variable]
    "latent_length": int,              # actual length T
}
```

**Text embedding file** (`text_embeddings/{id}.pt`):
```python
{
    "clap_embedding": torch.Tensor,    # [512]
    "clap_last_hidden": torch.Tensor,  # [clap_len, 512] - NOT USED
    "clap_len": int,
    "t5_last_hidden": torch.Tensor,    # [t5_len, 1024]
    "t5_len": int,
}
```

**Silence latent file** (`silence_10s_dacvae.pt`):
```python
{
    "posterior_params": torch.Tensor,  # [256, ~250]
    "latent_length": int,
}
```

### Verification Steps

1. **Unit test - file discovery:**
   ```python
   dataset = MultiSourceAudioDataset(
       dataset_roots=[".../WavCaps/FreeSound"],
       silence_latent_path=".../silence_10s_dacvae.pt",
   )
   print(f"Found {len(dataset)} samples")
   assert len(dataset) > 0
   ```

2. **Unit test - single item shapes:**
   ```python
   moments, prompt_data = dataset[0]
   assert moments.shape == (251, 256)
   assert prompt_data["clap"].shape == (512,)
   assert prompt_data["t5"].shape == (68, 1024)
   assert isinstance(prompt_data["t5_len"], int)
   ```

3. **Unit test - batch shapes:**
   ```python
   loader = DataLoader(dataset, batch_size=4, collate_fn=audio_collate_fn)
   moments, prompt_data = next(iter(loader))
   assert moments.shape == (4, 251, 256)
   assert prompt_data["clap"].shape == (4, 512)
   assert prompt_data["t5"].shape == (4, 68, 1024)
   assert prompt_data["t5_len"].shape == (4,)
   ```

4. **Unit test - padding/cropping:**
   - Find a short audio sample (< 251), verify silence padding works
   - Find a long audio sample (> 251), verify random crop produces valid output

5. **Integration test - full dataset:**
   ```python
   all_roots = [...]  # all 5 dataset roots
   dataset = MultiSourceAudioDataset(all_roots, silence_path)
   print(f"Total samples: {len(dataset)}")  # Should be ~400k+
   ```

---

# Step 2: Embeddings - Detailed Plan

## File: `src/models/modules/embeddings.py`

### Current State
```python
class ContinuousEmbedder(nn.Module):
    # Takes [B, D] → [B, hidden_size]
    # Single null_embedding [hidden_size] for CFG dropout
    # Replaces entire sample with null when dropped
```

### New Class: `SequencePromptEmbedder`

This class handles the 69-token prompt sequence (1 CLAP + 68 T5).

```python
class SequencePromptEmbedder(nn.Module):
    """
    Embeds multi-modal prompt (pooled CLAP + T5 hidden states) into
    a fixed-length sequence for conditioning.

    Output: [B, prompt_seq_len, hidden_size] where prompt_seq_len=69

    Token layout:
        Position 0: Projected pooled CLAP embedding
        Positions 1-68: Projected T5 hidden states (or null if padded)

    CFG Dropout: Replaces ALL 69 tokens with learned null embeddings.
    Padding: Positions beyond t5_len use null embeddings (same as CFG null).
    """

    def __init__(
        self,
        clap_dim: int,              # 512
        t5_dim: int,                # 1024
        hidden_size: int,           # Model hidden dim (e.g., 1536 for Large)
        prompt_seq_len: int = 69,   # Fixed output sequence length
        dropout_prob: float = 0.1,  # CFG dropout probability
    ) -> None:
        super().__init__()
        self.clap_dim = clap_dim
        self.t5_dim = t5_dim
        self.hidden_size = hidden_size
        self.prompt_seq_len = prompt_seq_len
        self.max_t5_tokens = prompt_seq_len - 1  # 68
        self.dropout_prob = dropout_prob

        # Separate projection layers for each modality
        self.clap_proj = nn.Linear(clap_dim, hidden_size, bias=False)
        self.t5_proj = nn.Linear(t5_dim, hidden_size, bias=False)

        # Learnable null embeddings - one per position in sequence
        # Used for: (1) padding short T5 sequences, (2) CFG dropout
        self.null_embeddings = nn.Parameter(torch.empty(prompt_seq_len, hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.clap_proj.weight, std=0.02)
        nn.init.normal_(self.t5_proj.weight, std=0.02)
        # CRITICAL: Non-zero init for null embeddings (zero causes NaN grads)
        nn.init.normal_(self.null_embeddings, std=0.02)
```

### Forward Method

```python
def forward(
    self,
    prompt_data: dict,
    train: bool,
    force_drop_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        prompt_data: dict containing:
            "clap": [B, clap_dim] pooled CLAP embeddings
            "t5": [B, max_t5_tokens, t5_dim] T5 hidden states (zero-padded)
            "t5_len": [B] actual T5 lengths (LongTensor)
        train: whether in training mode (enables random dropout)
        force_drop_ids: [B] tensor where 1 = force null embedding for that sample

    Returns:
        [B, prompt_seq_len, hidden_size] prompt sequence embeddings
    """
    clap = prompt_data["clap"]      # [B, 512]
    t5 = prompt_data["t5"]          # [B, 68, 1024]
    t5_len = prompt_data["t5_len"]  # [B]

    batch_size = clap.shape[0]
    device = clap.device
    dtype = clap.dtype

    # ===== Step 1: Project CLAP =====
    clap_emb = self.clap_proj(clap)  # [B, hidden_size]

    # ===== Step 2: Project T5 =====
    t5_emb = self.t5_proj(t5)  # [B, 68, hidden_size]

    # ===== Step 3: Create T5 padding mask =====
    # True = valid T5 token, False = padding position
    t5_positions = torch.arange(self.max_t5_tokens, device=device)
    t5_mask = t5_positions.unsqueeze(0) < t5_len.unsqueeze(1)  # [B, 68]

    # ===== Step 4: Replace padded T5 positions with null embeddings =====
    # null_embeddings[1:] are for T5 positions (index 0 is for CLAP)
    null_t5 = self.null_embeddings[1:].unsqueeze(0).expand(batch_size, -1, -1)  # [B, 68, H]
    t5_emb = torch.where(
        t5_mask.unsqueeze(-1),  # [B, 68, 1]
        t5_emb,
        null_t5,
    )

    # ===== Step 5: Concatenate CLAP + T5 =====
    # [B, 1, H] + [B, 68, H] -> [B, 69, H]
    prompt_seq = torch.cat([clap_emb.unsqueeze(1), t5_emb], dim=1)

    # ===== Step 6: CFG Dropout =====
    # Replace ENTIRE sequence with null embeddings for dropped samples
    use_dropout = self.dropout_prob > 0
    if (train and use_dropout) or (force_drop_ids is not None):
        if force_drop_ids is None:
            # Random dropout during training
            drop_ids = torch.rand(batch_size, device=device) < self.dropout_prob
        else:
            # Forced dropout (for CFG inference)
            drop_ids = force_drop_ids == 1

        # Full null sequence for dropped samples
        null_seq = self.null_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 69, H]

        prompt_seq = torch.where(
            drop_ids.view(-1, 1, 1),  # [B, 1, 1]
            null_seq,
            prompt_seq,
        )

    # ===== Step 7: Handle autocast =====
    if torch.is_autocast_enabled():
        prompt_seq = prompt_seq.to(torch.get_autocast_gpu_dtype())

    return prompt_seq  # [B, 69, hidden_size]
```

### Design Decisions

1. **Why separate projection layers?**
   - CLAP (512-dim) and T5 (1024-dim) have different dimensions and semantic spaces
   - Separate projections allow independent learning of how to map each modality

2. **Why 69 null embeddings instead of 1?**
   - Each position can learn its own "absence" representation
   - More expressive than broadcasting a single null
   - Matches the sequence structure (position 0 = CLAP null, positions 1-68 = T5 null)

3. **Why normal init for null embeddings?**
   - Zero initialization causes NaN gradients (observed in original ContinuousEmbedder)
   - std=0.02 matches the projection layer init

4. **Padding vs CFG dropout:**
   - Padding (short T5): Only positions beyond t5_len get null, CLAP stays real
   - CFG dropout: ALL 69 positions get null (enables unconditional generation)

### Integration Points

Models will need to change from:
```python
# Old: single token output
self.prompt_embedder = ContinuousEmbedder(conditioning_dim, hidden_size, dropout)
# Usage: label_emb = self.prompt_embedder(prompt, training)  # [B, H]
```

To:
```python
# New: sequence output
self.prompt_embedder = SequencePromptEmbedder(
    clap_dim=512,
    t5_dim=1024,
    hidden_size=hidden_size,
    prompt_seq_len=69,
    dropout_prob=dropout,
)
# Usage: prompt_seq = self.prompt_embedder(prompt_data, training)  # [B, 69, H]
```

### Verification Steps

1. **Shape test:**
   ```python
   embedder = SequencePromptEmbedder(
       clap_dim=512, t5_dim=1024, hidden_size=768, prompt_seq_len=69
   )
   prompt_data = {
       "clap": torch.randn(4, 512),
       "t5": torch.randn(4, 68, 1024),
       "t5_len": torch.tensor([10, 68, 5, 30]),
   }
   out = embedder(prompt_data, train=True)
   assert out.shape == (4, 69, 768)
   ```

2. **Padding behavior test:**
   ```python
   # Sample with t5_len=5 should have null at positions 6-68
   prompt_data = {
       "clap": torch.randn(1, 512),
       "t5": torch.randn(1, 68, 1024),
       "t5_len": torch.tensor([5]),
   }
   out = embedder(prompt_data, train=False)
   # Positions 6-68 (indices 6:69) should equal null_embeddings[6:69]
   # (after projection of zeros vs null - they should differ)
   ```

3. **CFG dropout test:**
   ```python
   force_drop = torch.tensor([1])  # Force drop
   out_dropped = embedder(prompt_data, train=False, force_drop_ids=force_drop)
   # Entire sequence should be null_embeddings
   expected = embedder.null_embeddings.unsqueeze(0)
   assert torch.allclose(out_dropped, expected)
   ```

4. **Gradient flow test:**
   ```python
   out = embedder(prompt_data, train=True)
   loss = out.sum()
   loss.backward()
   # Check no NaN gradients
   assert not torch.isnan(embedder.clap_proj.weight.grad).any()
   assert not torch.isnan(embedder.t5_proj.weight.grad).any()
   assert not torch.isnan(embedder.null_embeddings.grad).any()
   ```

5. **Dropout rate test:**
   ```python
   # Run many forward passes, check ~10% get dropped
   embedder.dropout_prob = 0.1
   drops = 0
   trials = 1000
   null_seq = embedder.null_embeddings.unsqueeze(0)
   for _ in range(trials):
       out = embedder(prompt_data, train=True)
       if torch.allclose(out, null_seq):
           drops += 1
   assert 0.05 < drops / trials < 0.15  # Should be ~10%
   ```

---

# Step 3: Models - Detailed Plan

## Overview

All 4 models share the same pattern:
1. **Current**: `prompt_embedder` outputs [B, hidden_size] → prepend 1 token
2. **New**: `prompt_embedder` outputs [B, 69, hidden_size] → prepend 69 tokens

Key change: **1 token → 69 tokens** everywhere prompt is used.

---

## Common Changes (All Models)

### Import Change
```python
# Old
from .modules.embeddings import TimestepEmbedder, LabelEmbedder, ContinuousEmbedder

# New
from .modules.embeddings import TimestepEmbedder, LabelEmbedder, ContinuousEmbedder, SequencePromptEmbedder
```

### Constructor Change
```python
# Old
if conditioning_type == "continuous":
    self.prompt_embedder = ContinuousEmbedder(conditioning_dim, hidden_size, class_dropout_prob)

# New
if conditioning_type == "continuous":
    assert conditioning_dim is not None
    # For backward compat, check if new params provided
    clap_dim = kwargs.get("clap_dim", 512)
    t5_dim = kwargs.get("t5_dim", 1024)
    prompt_seq_len = kwargs.get("prompt_seq_len", 69)
    self.prompt_embedder = SequencePromptEmbedder(
        clap_dim=clap_dim,
        t5_dim=t5_dim,
        hidden_size=hidden_size,
        prompt_seq_len=prompt_seq_len,
        dropout_prob=class_dropout_prob,
    )
    self.prompt_seq_len = prompt_seq_len
```

### Store prompt_seq_len
All models need to store `self.prompt_seq_len = 69` to know how many tokens to remove after blocks.

---

## Model 1: MaskedAR.py

### File: `src/models/MaskedAR.py`

### `forward_backbone` (lines 230-261)

**Current:**
```python
label_emb = self.prompt_embedder(prompt, self.training)
hidden_states = torch.cat((label_emb.unsqueeze(1), hidden_states), dim=1)  # [B, T+1, H]
# ... blocks ...
hidden_states = hidden_states[:, 1:, :]  # Remove 1 token
```

**New:**
```python
# prompt is now dict, output is [B, 69, H]
prompt_seq = self.prompt_embedder(prompt, self.training)
hidden_states = torch.cat((prompt_seq, hidden_states), dim=1)  # [B, T+69, H]
# ... blocks ...
hidden_states = hidden_states[:, self.prompt_seq_len:, :]  # Remove 69 tokens
```

### `forward_recurrent` (lines 263-311)

**Current (start_pos=0):**
```python
if start_pos == 0:
    hidden_states = self.prompt_embedder(
        hidden_states,  # This is prompt at pos 0
        self.training,
        force_drop_ids=prompt_drop_ids,
    ).unsqueeze(1)  # [B, 1, H]
```

**New (start_pos=0):**
```python
if start_pos == 0:
    # hidden_states is prompt_data dict at pos 0
    prompt_seq = self.prompt_embedder(
        hidden_states,
        self.training,
        force_drop_ids=prompt_drop_ids,
    )  # [B, 69, H]
    hidden_states = prompt_seq  # Already has seq dim
```

**Also need to track**: When `start_pos < self.prompt_seq_len`, we're still processing prompt tokens.

### `sample_with_cfg` (lines 334-416)

**Current CFG prompt handling:**
```python
# continuous conditioning
prompt = torch.cat([prompt, prompt], dim=0)
prompt_drop_ids = torch.zeros(prompt.shape[0], device=self.device, dtype=torch.long)
prompt_drop_ids[prompt.shape[0] // 2:] = 1
```

**New CFG prompt handling:**
```python
# prompt is dict with 'clap', 't5', 't5_len'
batch_size = prompt["clap"].shape[0]
combined_prompt = {
    "clap": torch.cat([prompt["clap"], prompt["clap"]], dim=0),
    "t5": torch.cat([prompt["t5"], prompt["t5"]], dim=0),
    "t5_len": torch.cat([prompt["t5_len"], prompt["t5_len"]], dim=0),
}
prompt_drop_ids = torch.zeros(batch_size * 2, device=self.device, dtype=torch.long)
prompt_drop_ids[batch_size:] = 1  # Second half gets null

# Pass combined_prompt to forward_recurrent
```

### AR sampling loop adjustment

**Current:** Loop `for i in range(self.seq_len)` with `start_pos=i`

**New:** Loop `for i in range(self.prompt_seq_len + self.seq_len)` or handle prompt positions specially.

Actually, cleaner approach: Process all 69 prompt tokens at once at position 0, then continue from position 69.

```python
# First pass: all 69 prompt tokens at once
conditioning = self.forward_recurrent(
    prompt,  # dict
    start_pos=0,
    inference_params=inference_params,
    append_mask=True,
    prompt_drop_ids=prompt_drop_ids,
)
conditioning = conditioning[:, -1:]  # Get the mask position output

# Subsequent passes: same as before, but start_pos begins at 69
for i in range(self.seq_len):
    # ...
    conditioning = self.forward_recurrent(
        recurrent_input,
        start_pos=self.prompt_seq_len + i,  # Offset by 69
        # ...
    )
```

---

## Model 2: AR_DiT.py

### File: `src/models/AR_DiT.py`

### `forward` (lines 159-215)

**Current:**
```python
label_emb = self.prompt_embedder(prompt, self.training, force_drop_ids=prompt_drop_ids)  # [B, H]
# Zero time modulation for single class token
t_cls_mod = torch.zeros(batch_size, 1, time_modulation.size(-1), ...)
time_modulation = torch.cat([t_cls_mod, time_modulation], dim=1)  # [B, T+1, 6H]
hidden_states = torch.cat([label_emb.unsqueeze(1), hidden_states], dim=1)  # [B, T+1, H]
# ... blocks ...
hidden_states = hidden_states[:, 1:, :]  # Remove 1 token
```

**New:**
```python
prompt_seq = self.prompt_embedder(prompt, self.training, force_drop_ids=prompt_drop_ids)  # [B, 69, H]
# Zero time modulation for all 69 prompt tokens
t_prompt_mod = torch.zeros(batch_size, self.prompt_seq_len, time_modulation.size(-1), ...)
time_modulation = torch.cat([t_prompt_mod, time_modulation], dim=1)  # [B, T+69, 6H]
hidden_states = torch.cat([prompt_seq, hidden_states], dim=1)  # [B, T+69, H]
# ... blocks ...
hidden_states = hidden_states[:, self.prompt_seq_len:, :]  # Remove 69 tokens
```

### `sample_with_cfg` (lines 217-255)

Same dict-based CFG handling as MaskedAR.

---

## Model 3: DiT.py

### File: `src/models/DiT.py`

### `forward` (lines 146-169)

**Current:**
```python
label_emb = self.prompt_embedder(prompt, self.training, force_drop_ids=prompt_drop_ids)
hidden_states = torch.cat([label_emb.unsqueeze(1), hidden_states], dim=1)  # [B, T+1, H]
# ... blocks ...
hidden_states = hidden_states[:, 1:, :]  # Remove 1 token
```

**New:**
```python
prompt_seq = self.prompt_embedder(prompt, self.training, force_drop_ids=prompt_drop_ids)  # [B, 69, H]
hidden_states = torch.cat([prompt_seq, hidden_states], dim=1)  # [B, T+69, H]
# ... blocks (bidirectional attention) ...
hidden_states = hidden_states[:, self.prompt_seq_len:, :]  # Remove 69 tokens
```

### `sample_with_cfg` (lines 171-209)

Same dict-based CFG handling.

---

## Model 4: Transformer.py

### File: `src/models/Transformer.py`

### `forward_parallel` (lines 215-231)

**Current:**
```python
label_emb = self.prompt_embedder(prompt, self.training, force_drop_ids=prompt_drop_ids)
hidden_states = torch.cat((label_emb.unsqueeze(1), hidden_states[:, :-1]), dim=1)  # Shift by 1
```

**New:**
```python
prompt_seq = self.prompt_embedder(prompt, self.training, force_drop_ids=prompt_drop_ids)  # [B, 69, H]
# Shift input: drop last 69 tokens to make room for prompt
hidden_states = torch.cat((prompt_seq, hidden_states[:, :-self.prompt_seq_len]), dim=1)
```

### `forward_recurrent` (lines 233-257)

**Current (start_pos=0):**
```python
if start_pos == 0:
    hidden_states = self.prompt_embedder(hidden_states, self.training, ...).unsqueeze(1)
```

**New (start_pos=0):**
```python
if start_pos == 0:
    # Process all 69 prompt tokens at once
    hidden_states = self.prompt_embedder(hidden_states, self.training, ...)  # [B, 69, H]
```

### `sample_with_cfg` (lines 276-321)

Same dict-based CFG handling, plus adjustment for AR loop positions.

---

## Helper Function for CFG Prompt Handling

Add to each model or as utility:

```python
def _prepare_cfg_prompt(self, prompt: dict) -> tuple[dict, torch.Tensor]:
    """
    Prepare prompt dict for classifier-free guidance.
    Returns (combined_prompt, prompt_drop_ids) where:
    - combined_prompt has 2*B batch size (cond + uncond)
    - prompt_drop_ids marks second half for null embedding
    """
    batch_size = prompt["clap"].shape[0]
    combined_prompt = {
        "clap": torch.cat([prompt["clap"], prompt["clap"]], dim=0),
        "t5": torch.cat([prompt["t5"], prompt["t5"]], dim=0),
        "t5_len": torch.cat([prompt["t5_len"], prompt["t5_len"]], dim=0),
    }
    prompt_drop_ids = torch.zeros(batch_size * 2, device=self.device, dtype=torch.long)
    prompt_drop_ids[batch_size:] = 1
    return combined_prompt, prompt_drop_ids
```

---

## Constructor Parameter Additions

All models need new kwargs in their config functions:

```python
def MaskedAR_L(**kwargs) -> MaskedARTransformer:
    return MaskedARTransformer(
        depth=32,
        hidden_size=1024,
        # ... existing params ...
        **kwargs,  # Passes through clap_dim, t5_dim, prompt_seq_len
    )
```

And the constructor signature changes:

```python
def __init__(
    self,
    seq_len: int = 1024,
    # ... existing params ...
    clap_dim: int = 512,        # NEW
    t5_dim: int = 1024,         # NEW
    prompt_seq_len: int = 69,   # NEW
) -> None:
```

---

## Verification Steps

1. **Shape verification (each model):**
   ```python
   model = MaskedARTransformer(
       seq_len=251,
       in_channels=128,
       conditioning_type="continuous",
       conditioning_dim=512,
       clap_dim=512,
       t5_dim=1024,
       prompt_seq_len=69,
   )
   prompt_data = {
       "clap": torch.randn(2, 512),
       "t5": torch.randn(2, 68, 1024),
       "t5_len": torch.tensor([30, 50]),
   }
   # Test forward passes work with new prompt structure
   ```

2. **CFG verification:**
   ```python
   # Verify sample_with_cfg produces correct output shapes
   # Verify cond/uncond splitting works correctly
   ```

3. **Gradient flow:**
   ```python
   # Verify gradients flow through prompt_embedder
   ```

4. **Position encoding compatibility:**
   ```python
   # Verify RoPE handles longer sequences (T+69 instead of T+1)
   # May need to adjust max_seqlen in InferenceParams
   ```

---

## Summary of Changes Per Model

| Model | forward | forward_parallel | forward_recurrent | sample_with_cfg |
|-------|---------|-----------------|-------------------|-----------------|
| MaskedAR | N/A (calls backbone) | N/A | ✓ pos 0 → 69 tokens | ✓ dict handling |
| AR_DiT | ✓ 69 tokens + time mod | N/A | N/A | ✓ dict handling |
| DiT | ✓ 69 tokens | N/A | N/A | ✓ dict handling |
| Transformer | N/A (calls parallel) | ✓ shift by 69 | ✓ pos 0 → 69 tokens | ✓ dict handling |

---

# Step 4: Training Script - Detailed Plan

## Files to Modify
- `train_audio.py` - CLI arguments and datamodule setup
- `src/lightning.py` - LitModule constructor and training_step

---

## train_audio.py

### New/Modified Arguments

```python
# ===== Dataset Configuration (NEW) =====
p.add_argument("--wavcaps-root", type=str,
               default="/share/users/student/f/friverossego/datasets/WavCaps",
               help="Root directory for WavCaps dataset")
p.add_argument("--audiocaps-root", type=str,
               default="/share/users/student/f/friverossego/datasets/AudioCaps",
               help="Root directory for AudioCaps dataset")
p.add_argument("--silence-latent-path", type=str,
               default="silence_samples/silence_10s_dacvae.pt",
               help="Path to silence latent for padding")

# ===== Prompt Configuration (NEW) =====
p.add_argument("--clap-dim", type=int, default=512,
               help="CLAP pooled embedding dimension")
p.add_argument("--t5-dim", type=int, default=1024,
               help="T5 hidden state dimension")
p.add_argument("--prompt-seq-len", type=int, default=69,
               help="Fixed prompt sequence length (1 CLAP + 68 T5 max)")

# ===== Existing args to keep =====
p.add_argument("--seq-len", type=int, default=251,
               help="Audio sequence length (target for DACVAE latents)")
p.add_argument("--conditioning-type", type=str, default="continuous",
               help="Conditioning type: 'class' or 'continuous'")
# Note: --conditioning-dim is no longer used directly, kept for backward compat
```

### Build Dataset Roots Function

```python
def build_dataset_roots(args) -> list[str]:
    """Build list of dataset root paths for training."""
    roots = []

    # WavCaps subsets (all for training)
    wavcaps_subsets = ["AudioSet_SL", "BBC_Sound_Effects", "FreeSound", "SoundBible"]
    for subset in wavcaps_subsets:
        subset_path = os.path.join(args.wavcaps_root, subset)
        if os.path.exists(subset_path):
            roots.append(subset_path)
        else:
            print(f"Warning: WavCaps subset not found: {subset_path}")

    # AudioCaps train split only
    audiocaps_train = os.path.join(args.audiocaps_root, "train")
    if os.path.exists(audiocaps_train):
        roots.append(audiocaps_train)
    else:
        print(f"Warning: AudioCaps train not found: {audiocaps_train}")

    if len(roots) == 0:
        raise ValueError("No dataset roots found!")

    return roots
```

### Updated main() Function

```python
def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_everything(args.global_seed, workers=True)

    # Build dataset configuration
    dataset_roots = build_dataset_roots(args)

    # Resolve silence latent path (relative to script dir or absolute)
    silence_path = args.silence_latent_path
    if not os.path.isabs(silence_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        silence_path = os.path.join(script_dir, silence_path)

    # Create datamodule with new structure
    dm = CachedAudioDataModule(
        dataset_roots=dataset_roots,
        silence_latent_path=silence_path,
        batch_size=args.batch_size,
        target_seq_len=args.seq_len,
        max_t5_tokens=args.prompt_seq_len - 1,  # 68
        num_workers=args.num_workers,
    )

    # Create LitModule with new conditioning params
    lit = LitModule(
        model_name=args.model,
        seq_len=args.seq_len,
        latent_size=args.latent_size,
        num_classes=1,  # unused for continuous
        conditioning_type=args.conditioning_type,
        clap_dim=args.clap_dim,          # NEW
        t5_dim=args.t5_dim,              # NEW
        prompt_seq_len=args.prompt_seq_len,  # NEW
        prediction_type=args.prediction_type,
        batch_mul=args.batch_mul,
        mask_prob_min=args.mask_prob_min,
        mask_prob_max=args.mask_prob_max,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
    )

    # ... rest of training setup unchanged ...
```

---

## src/lightning.py

### Updated Constructor Signature

```python
class LitModule(L.LightningModule):
    def __init__(
        self,
        model_name: str = "Transformer-L",
        seq_len: int | None = None,
        input_size: int | None = None,
        latent_size: int = 16,
        num_classes: int = 1000,
        conditioning_type: str = "class",
        conditioning_dim: int | None = None,  # Keep for backward compat
        # ===== NEW PARAMS =====
        clap_dim: int = 512,
        t5_dim: int = 1024,
        prompt_seq_len: int = 69,
        # ======================
        prediction_type: str = "flow",
        t_m: float = 0.0,
        t_s: float = 1.0,
        batch_mul: int = 4,
        mask_prob_min: float = 0.5,
        mask_prob_max: float = 0.5,
        data_scale: float = 1.0,
        data_bias: float = 0.0,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        lr_scheduler: str = "cosine",
        lr_warmup_steps: int = 1000,
    ):
```

### Updated Model Creation

```python
        self.save_hyperparameters()

        # Build model kwargs for new conditioning
        model_kwargs = dict(
            seq_len=seq_len,
            in_channels=latent_size,
            num_classes=num_classes,
            conditioning_type=conditioning_type,
        )

        if conditioning_type == "continuous":
            # Use new sequence prompt embedder params
            model_kwargs.update(
                clap_dim=clap_dim,
                t5_dim=t5_dim,
                prompt_seq_len=prompt_seq_len,
                conditioning_dim=clap_dim,  # For any legacy code paths
            )
        else:
            # Class conditioning (unchanged)
            model_kwargs["conditioning_dim"] = conditioning_dim

        self.model = All_models[model_name](**model_kwargs)
```

### Updated training_step

```python
def training_step(self, batch, batch_idx):
    moments, prompt_data = batch
    # prompt_data is now dict: {"clap": [B, 512], "t5": [B, 68, 1024], "t5_len": [B]}

    posterior = SequenceDiagonalGaussianDistribution(moments)
    x0 = posterior.sample()
    x0 = self._normalize(x0)

    # Pass prompt_data dict to scheduler (which passes to model)
    loss = self.noise_scheduler.get_losses(self.model, x0, prompt_data)

    self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
    return loss
```

### Updated sample_latents

```python
@torch.no_grad()
def sample_latents(
    self,
    prompt,  # Now dict or tensor depending on conditioning_type
    cfg_scale: float = 4.0,
    num_inference_steps: int = 250,
    scheduler=None,
    ardiff_step: int = None,
    base_num_frames: int = None,
):
    self.eval()
    if scheduler is None:
        scheduler = self.noise_scheduler

    scheduler.configure_sampling(
        ardiff_step=ardiff_step,
        base_num_frames=base_num_frames,
    )
    scheduler.set_timesteps(num_inference_steps, device=self.device)

    # prompt is passed directly to model.sample_with_cfg
    # Model handles dict structure internally
    latents = self.model.sample_with_cfg(prompt, cfg_scale, scheduler)
    return self.unnormalize_latents(latents)
```

---

## Flow Matching Schedulers Update

### File: `src/flow_matching.py`

The schedulers pass `prompt` to model forward. No changes needed if models handle dict internally.

Check each scheduler's `get_losses` method:

```python
# FlowMatchingSchedulerMaskedAR.get_losses (line ~514)
def get_losses(self, model, x0_seq, prompt) -> torch.Tensor:
    # ...
    model_output = model(
        x_noisy,
        timesteps,
        x_start=x0_seq,
        prompt=prompt,  # This is now a dict - model handles it
        mask=mask,
        flat_mask_indices=flat_mask_indices,
        batch_mul=self.batch_mul,
    )
```

No changes needed in schedulers - they just pass prompt through.

---

## Verification Steps

1. **CLI argument parsing:**
   ```bash
   python train_audio.py --help
   # Verify new args appear
   ```

2. **Dataset root discovery:**
   ```python
   # Verify all 5 roots are found
   roots = build_dataset_roots(args)
   assert len(roots) == 5
   ```

3. **Single training step:**
   ```python
   dm = CachedAudioDataModule(...)
   dm.setup()
   batch = next(iter(dm.train_dataloader()))
   moments, prompt_data = batch

   # Verify shapes
   assert moments.shape == (batch_size, 251, 256)
   assert prompt_data["clap"].shape == (batch_size, 512)
   assert prompt_data["t5"].shape == (batch_size, 68, 1024)

   # Run training step
   lit = LitModule(...)
   loss = lit.training_step(batch, 0)
   assert not torch.isnan(loss)
   ```

4. **Full integration test:**
   ```bash
   python train_audio.py \
       --wavcaps-root /path/to/WavCaps \
       --audiocaps-root /path/to/AudioCaps \
       --batch-size 2 \
       --epochs 1 \
       --log-every 1
   # Verify training runs without errors
   ```

---

# Step 5: Inference - Detailed Plan

## File: `sample_audio.py`

### New Imports

```python
from transformers import AutoTokenizer, T5EncoderModel
```

### New Arguments

```python
parser.add_argument("--t5-model", type=str, default="google/flan-t5-large",
                    help="HuggingFace T5 model for text encoding")
parser.add_argument("--max-t5-tokens", type=int, default=68,
                    help="Maximum T5 tokens (prompt_seq_len - 1)")
parser.add_argument("--clap-dim", type=int, default=512,
                    help="CLAP pooled embedding dimension")
parser.add_argument("--t5-dim", type=int, default=1024,
                    help="T5 hidden state dimension")
```

### Load T5 Model Function

```python
def load_t5_model(model_id: str, device: torch.device):
    """Load T5 encoder model and tokenizer."""
    print(f"Loading T5 model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = T5EncoderModel.from_pretrained(model_id).eval().to(device)
    return model, tokenizer
```

### New Text Embedding Function

```python
@torch.no_grad()
def get_text_embeddings(
    clap_model,
    clap_processor,
    t5_model,
    t5_tokenizer,
    text: str,
    device: torch.device,
    max_t5_tokens: int = 68,
    clap_dim: int = 512,
    t5_dim: int = 1024,
) -> dict:
    """
    Generate CLAP pooled embedding and T5 hidden states from text.

    Returns:
        dict with:
            "clap": [1, clap_dim] pooled CLAP embedding
            "t5": [1, max_t5_tokens, t5_dim] T5 hidden states (padded/truncated)
            "t5_len": [1] actual T5 length
    """
    # ===== CLAP Embedding =====
    clap_inputs = clap_processor(
        text=[text],
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    clap_emb = clap_model.get_text_features(**clap_inputs)  # [1, 512]

    # ===== T5 Hidden States =====
    t5_inputs = t5_tokenizer(
        [text],
        padding=False,
        truncation=True,
        max_length=max_t5_tokens + 10,  # Allow some buffer for truncation
        return_tensors="pt"
    ).to(device)

    t5_output = t5_model(input_ids=t5_inputs["input_ids"], return_dict=True)
    t5_hidden = t5_output.last_hidden_state[0]  # [seq_len, hidden_dim]

    actual_len = t5_hidden.shape[0]

    # Truncate or pad T5 to max_t5_tokens
    if actual_len > max_t5_tokens:
        t5_hidden = t5_hidden[:max_t5_tokens]
        actual_len = max_t5_tokens
    elif actual_len < max_t5_tokens:
        pad_size = max_t5_tokens - actual_len
        padding = torch.zeros(pad_size, t5_dim, device=device, dtype=t5_hidden.dtype)
        t5_hidden = torch.cat([t5_hidden, padding], dim=0)

    return {
        "clap": clap_emb,  # [1, 512]
        "t5": t5_hidden.unsqueeze(0),  # [1, 68, 1024]
        "t5_len": torch.tensor([actual_len], device=device, dtype=torch.long),
    }
```

### Updated Main Function

```python
@torch.no_grad()
def main():
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_autocast_dtype(args.precision, device)
    autocast = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )

    # ===== Load Model =====
    print(f"Loading model from {args.checkpoint}")
    lit = LitModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    lit.to(device=device)
    lit.eval()

    # ===== Get Prompts =====
    if args.embedding is not None:
        # Load pre-computed embeddings (must be in new dict format)
        print(f"Loading embeddings from {args.embedding}")
        embeddings = torch.load(args.embedding, map_location=device)
        # Expect: {"clap": [N, 512], "t5": [N, 68, 1024], "t5_len": [N]}
        num_samples = embeddings["clap"].shape[0]
        prompts = [f"embedding_{i}" for i in range(num_samples)]
        all_prompt_data = [
            {
                "clap": embeddings["clap"][i:i+1],
                "t5": embeddings["t5"][i:i+1],
                "t5_len": embeddings["t5_len"][i:i+1],
            }
            for i in range(num_samples)
        ]
    else:
        # Generate embeddings from text
        if args.text is not None:
            prompts = [args.text]
        elif args.text_file is not None:
            with open(args.text_file, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            prompts = DEFAULT_PROMPTS

        # Load CLAP model
        print(f"Loading CLAP model: {args.clap_model}")
        clap_model, clap_processor = load_clap_model(args.clap_model, device)

        # Load T5 model
        print(f"Loading T5 model: {args.t5_model}")
        t5_model, t5_tokenizer = load_t5_model(args.t5_model, device)

        # Generate embeddings for all prompts
        print("Generating embeddings...")
        all_prompt_data = []
        for prompt in prompts:
            prompt_data = get_text_embeddings(
                clap_model, clap_processor,
                t5_model, t5_tokenizer,
                prompt, device,
                max_t5_tokens=args.max_t5_tokens,
                clap_dim=args.clap_dim,
                t5_dim=args.t5_dim,
            )
            all_prompt_data.append(prompt_data)

        # Free CLAP and T5 memory
        del clap_model, clap_processor, t5_model, t5_tokenizer
        torch.cuda.empty_cache()

    # ===== Load DACVAE for Decoding =====
    print("Loading DACVAE...")
    dacvae_model = load_dacvae(args.dacvae_weights, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ===== Generate Samples =====
    print(f"Generating {len(prompts)} audio samples...")
    for i, (prompt_text, prompt_data) in enumerate(tqdm(zip(prompts, all_prompt_data), total=len(prompts))):
        with autocast:
            latents = lit.sample_latents(
                prompt_data,  # dict with clap, t5, t5_len
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
            )

        # latents shape: [1, T, D] -> need [1, D, T] for DACVAE
        latents = latents.transpose(1, 2)

        # Decode with DACVAE
        metadata = {
            "sample_rate": args.sample_rate,
            "latent_length": latents.shape[-1],
        }
        with torch.autocast(device_type=device.type, enabled=False):
            audio = decode_audio_latents(dacvae_model, latents.float(), metadata)

        # Save audio
        audio = audio.detach().cpu()
        if audio.dim() == 3:
            audio = audio[0]  # Remove batch dim

        # Create safe filename from prompt
        if isinstance(prompt_text, str) and not prompt_text.startswith("embedding_"):
            safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt_text)[:50]
        else:
            safe_name = prompt_text
        filename = f"{i:04d}_{safe_name}.mp3"
        filepath = os.path.join(args.output_dir, filename)

        torchaudio.save(filepath, audio, args.sample_rate, format="mp3")
        print(f"Saved: {filepath}")

    print(f"\nGenerated {len(prompts)} audio samples in {args.output_dir}")
```

### Pre-compute Embeddings Script (Optional)

For batch inference, you might want a script to pre-compute embeddings:

```python
# precompute_embeddings.py
def precompute_embeddings(
    text_file: str,
    output_file: str,
    clap_model_id: str,
    t5_model_id: str,
    max_t5_tokens: int = 68,
):
    """Pre-compute CLAP + T5 embeddings for a list of prompts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    clap_model, clap_processor = load_clap_model(clap_model_id, device)
    t5_model, t5_tokenizer = load_t5_model(t5_model_id, device)

    # Load prompts
    with open(text_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Generate embeddings
    all_clap = []
    all_t5 = []
    all_t5_len = []

    for prompt in tqdm(prompts):
        data = get_text_embeddings(
            clap_model, clap_processor,
            t5_model, t5_tokenizer,
            prompt, device, max_t5_tokens,
        )
        all_clap.append(data["clap"])
        all_t5.append(data["t5"])
        all_t5_len.append(data["t5_len"])

    # Save
    torch.save({
        "clap": torch.cat(all_clap, dim=0),
        "t5": torch.cat(all_t5, dim=0),
        "t5_len": torch.cat(all_t5_len, dim=0),
        "prompts": prompts,
    }, output_file)
```

---

## Verification Steps

1. **T5 model loading:**
   ```python
   t5_model, t5_tokenizer = load_t5_model("google/flan-t5-large", device)
   # Verify loads without errors
   ```

2. **Text embedding generation:**
   ```python
   prompt_data = get_text_embeddings(
       clap_model, clap_processor,
       t5_model, t5_tokenizer,
       "A dog barking loudly",
       device,
   )
   assert prompt_data["clap"].shape == (1, 512)
   assert prompt_data["t5"].shape == (1, 68, 1024)
   assert prompt_data["t5_len"].shape == (1,)
   ```

3. **Full inference pipeline:**
   ```bash
   python sample_audio.py \
       --checkpoint path/to/checkpoint.ckpt \
       --text "A dog barking in the distance" \
       --output-dir test_samples
   # Verify audio file is generated
   ```

4. **Batch inference with pre-computed embeddings:**
   ```bash
   python sample_audio.py \
       --checkpoint path/to/checkpoint.ckpt \
       --embedding path/to/embeddings.pt \
       --output-dir batch_samples
   ```

5. **CFG verification:**
   ```python
   # Generate with different CFG scales, verify quality differences
   for cfg in [1.0, 4.0, 7.0]:
       latents = lit.sample_latents(prompt_data, cfg_scale=cfg)
   ```
