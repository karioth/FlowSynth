# Audio-only Vision Removal Plan

## Decisions
- Conditioning: continuous-only prompts (remove class-conditioning paths).
- Compatibility: drop legacy (remove input_size fallback and label_embedder remap).
- Assets: remove vision weights/artifacts.

## Inventory (vision-specific)
- Entry points: `train_vision.py`, `preprocess_vision.py`, `sample_vision.py`, `evaluate_vision.py`
- Data utils: `src/data_utils/vision_datamodule.py`, `src/data_utils/preprocess/img_items.py`
- Vision VAE: `src/models/modules/vae.py`
- Metrics: `src/metrics/__init__.py`, `src/metrics/fid.py`, `src/metrics/inception.py`, `src/metrics/IS.py`
- Assets: `pretrained_models/kl16.ckpt`
- Docs: `README.md`, `AGENTS.md`
- Audio scripts needing signature updates: `train_audio.py`, `train_audio.sh`, `train_audio_deniz.sh`, `overfit_sanity.py`
- Shared model code to simplify: `src/lightning.py`, `src/models/DiT.py`, `src/models/Transformer.py`, `src/models/AR_DiT.py`, `src/models/MaskedAR.py`, `src/models/modules/embeddings.py`

## Plan
1. Remove vision-only entry points and helpers
   - Delete `train_vision.py`, `preprocess_vision.py`, `sample_vision.py`, `evaluate_vision.py`.
   - Delete `src/data_utils/vision_datamodule.py`.
   - Delete `src/data_utils/preprocess/img_items.py`.

2. Remove vision metrics package
   - Delete `src/metrics/__init__.py`, `src/metrics/fid.py`, `src/metrics/inception.py`, `src/metrics/IS.py`.

3. Remove vision VAE module + assets
   - Delete `src/models/modules/vae.py`.
   - Remove `pretrained_models/kl16.ckpt` (vision-only weight).

4. Simplify shared models to continuous-only prompting
   - `src/models/modules/embeddings.py`: remove `LabelEmbedder`; keep `SequencePromptEmbedder`.
   - `src/models/DiT.py`, `src/models/Transformer.py`, `src/models/AR_DiT.py`, `src/models/MaskedAR.py`:
     - Remove `conditioning_type`, `num_classes`, and `conditioning_dim` arguments.
     - Replace class/continuous branches with a single prompt-dict path.
     - Remove class-specific CFG logic (label null tokens); keep prompt-drop CFG using prompt dict.
     - Rename `class_dropout_prob` -> `prompt_dropout_prob` for clarity.
     - Update `__main__` shape checks to new signatures.

5. Simplify Lightning module API
   - `src/lightning.py`:
     - Remove `input_size` fallback; require `seq_len`.
     - Remove legacy `label_embedder` remap in `load_state_dict`.
     - Remove `conditioning_type`, `conditioning_dim`, and `num_classes` params.
     - Optional: drop `data_scale`/`data_bias` normalization to keep audio latents unmodified.

6. Update audio entry points and scripts
   - `train_audio.py`: remove `--conditioning-type` and any class-related args; align with new LitModule/model signatures.
   - `train_audio.sh`, `train_audio_deniz.sh`: drop `--conditioning-type` flags.
   - `overfit_sanity.py`: remove class-conditioning args and update model constructors.

7. Docs and dependency cleanup
   - `README.md`: remove vision pipeline sections and metrics; keep audio-only instructions.
   - `AGENTS.md`: remove vision references and drop vision-only deps (torchvision, PIL, scipy, requests, torch_fidelity).

8. Validation
   - Search for leftovers: `rg "vision|imagenet|FID|inception|torchvision|kl16|ImageNet"` from repo root.
   - Quick import checks: `python train_audio.py --help`, `python sample_audio.py --help`, `python preprocess_audio.py --help`.
   - Optional: `python -m compileall src` to catch syntax/import errors.

## Compatibility Notes
- Existing vision checkpoints will no longer load.
- Existing audio checkpoints that stored legacy hparams (`conditioning_type`, `input_size`, class embedder names) will not load; re-export/retrain is required.
