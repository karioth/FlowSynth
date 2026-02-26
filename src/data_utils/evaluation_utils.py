"""
Utilities extracted from evaluate.py to keep evaluation orchestration clean.
"""

from __future__ import annotations

import contextlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from kadtk.fad import calc_frechet_distance
from kadtk.kad import calc_kernel_audio_distance
from kadtk.model_loader import (
    CLAPLaionModel,
    CLAPModel,
    CdpamModel,
    DACModel,
    EncodecEmbModel,
    HuBERTModel,
    MERTModel,
    ModelLoader,
    PANNsModel,
    PaSSTModel,
    VGGishModel,
    W2V2Model,
    WavLMModel,
    WhisperModel,
)

ModelFactory = Callable[[Optional[float]], ModelLoader]
ModelRegistry = dict[str, ModelFactory]


def configure_numba_cache_dir(preferred_dir: Optional[str] = None) -> str:
    """
    Route numba cache to a user-writable location.
    This avoids laion_clap defaulting NUMBA cache to /tmp (can be quota-limited).
    """
    cache_dir = preferred_dir
    if cache_dir is None:
        cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    if cache_dir and cache_dir.rstrip("/").startswith("/tmp"):
        # Treat /tmp cache targets as invalid for our environment.
        cache_dir = None
    if not cache_dir:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            cache_dir = os.path.join(xdg_cache_home, "numba")
        else:
            cache_dir = os.path.join(str(Path.home()), ".cache", "numba")

    os.makedirs(cache_dir, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = cache_dir
    try:
        import numba  # pylint: disable=import-outside-toplevel

        numba.config.CACHE_DIR = cache_dir
    except Exception:
        # numba may not be imported yet in this process.
        pass
    return cache_dir


def build_model_registry() -> ModelRegistry:
    registry: ModelRegistry = {
        "clap-2023": lambda audio_len: CLAPModel("2023", audio_len=audio_len),
        "clap-laion-audio": lambda audio_len: CLAPLaionModel("audio", audio_len=audio_len),
        "clap-laion-music": lambda audio_len: CLAPLaionModel("music", audio_len=audio_len),
        "vggish": lambda audio_len: VGGishModel(audio_len=audio_len),
        "panns-cnn14-32k": lambda audio_len: PANNsModel("cnn14-32k", audio_len=audio_len),
        "panns-cnn14-16k": lambda audio_len: PANNsModel("cnn14-16k", audio_len=audio_len),
        "panns-wavegram-logmel": lambda audio_len: PANNsModel("wavegram-logmel", audio_len=audio_len),
        "encodec-emb": lambda audio_len: EncodecEmbModel("24k", audio_len=audio_len),
        "encodec-emb-48k": lambda audio_len: EncodecEmbModel("48k", audio_len=audio_len),
        "dac-44kHz": lambda audio_len: DACModel(audio_len=audio_len),
        "cdpam-acoustic": lambda audio_len: CdpamModel("acoustic", audio_len=audio_len),
        "cdpam-content": lambda audio_len: CdpamModel("content", audio_len=audio_len),
        "whisper-tiny": lambda audio_len: WhisperModel("tiny", audio_len=audio_len),
        "whisper-small": lambda audio_len: WhisperModel("small", audio_len=audio_len),
        "whisper-base": lambda audio_len: WhisperModel("base", audio_len=audio_len),
        "whisper-medium": lambda audio_len: WhisperModel("medium", audio_len=audio_len),
        "whisper-large": lambda audio_len: WhisperModel("large", audio_len=audio_len),
        "passt-base-10s": lambda audio_len: PaSSTModel("base-10s", audio_len=audio_len),
        "passt-base-20s": lambda audio_len: PaSSTModel("base-20s", audio_len=audio_len),
        "passt-base-30s": lambda audio_len: PaSSTModel("base-30s", audio_len=audio_len),
        "passt-openmic": lambda audio_len: PaSSTModel("openmic", audio_len=audio_len),
        "passt-fsd50k": lambda audio_len: PaSSTModel("fsd50k", audio_len=audio_len),
    }

    for layer in range(1, 13):
        model_name = f"MERT-v1-95M-{layer}" if layer != 12 else "MERT-v1-95M"
        registry[model_name] = lambda audio_len, _layer=layer: MERTModel("v1-95M", layer=_layer, audio_len=audio_len)

    for layer in range(1, 13):
        model_name = f"w2v2-base-{layer}" if layer != 12 else "w2v2-base"
        registry[model_name] = lambda audio_len, _layer=layer: W2V2Model("base", layer=_layer, audio_len=audio_len)

    for layer in range(1, 25):
        model_name = f"w2v2-large-{layer}" if layer != 24 else "w2v2-large"
        registry[model_name] = lambda audio_len, _layer=layer: W2V2Model("large", layer=_layer, audio_len=audio_len)

    for layer in range(1, 13):
        model_name = f"hubert-base-{layer}" if layer != 12 else "hubert-base"
        registry[model_name] = lambda audio_len, _layer=layer: HuBERTModel("base", layer=_layer, audio_len=audio_len)

    for layer in range(1, 25):
        model_name = f"hubert-large-{layer}" if layer != 24 else "hubert-large"
        registry[model_name] = lambda audio_len, _layer=layer: HuBERTModel("large", layer=_layer, audio_len=audio_len)

    for layer in range(1, 13):
        model_name = f"wavlm-base-{layer}" if layer != 12 else "wavlm-base"
        registry[model_name] = lambda audio_len, _layer=layer: WavLMModel("base", layer=_layer, audio_len=audio_len)

    for layer in range(1, 13):
        model_name = f"wavlm-base-plus-{layer}" if layer != 12 else "wavlm-base-plus"
        registry[model_name] = lambda audio_len, _layer=layer: WavLMModel("base-plus", layer=_layer, audio_len=audio_len)

    for layer in range(1, 25):
        model_name = f"wavlm-large-{layer}" if layer != 24 else "wavlm-large"
        registry[model_name] = lambda audio_len, _layer=layer: WavLMModel("large", layer=_layer, audio_len=audio_len)

    return registry


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"CUDA device requested ({device_arg}) but CUDA is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise SystemExit(
                f"Invalid CUDA device index {device.index}; found {torch.cuda.device_count()} CUDA device(s)."
            )
    return device


def prepare_model_for_inference(model: ModelLoader, device: torch.device) -> None:
    """
    Configure model device semantics (including CLAP quirk) and load weights.
    """
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device.index)
        # CLAPModel's internal CUDA toggle checks equality against torch.device('cuda').
        if isinstance(model, CLAPModel):
            model.device = torch.device("cuda")
        else:
            model.device = device
    else:
        model.device = torch.device("cpu")

    preferred_numba_cache: Optional[str] = None
    if isinstance(model, CLAPLaionModel):
        # laion_clap import side effects can overwrite NUMBA cache to /tmp.
        preferred_numba_cache = configure_numba_cache_dir()
        try:
            import laion_clap  # pylint: disable=import-outside-toplevel, unused-import
        except Exception:
            pass
        configure_numba_cache_dir(preferred_numba_cache)

    suppress_noisy_load_logs = isinstance(model, (CLAPLaionModel, PaSSTModel))
    with torch.no_grad():
        if preferred_numba_cache is not None:
            configure_numba_cache_dir(preferred_numba_cache)
        if suppress_noisy_load_logs:
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    model.load_model()
        else:
            model.load_model()


def collect_wavs(audio_dir: Path, limit: int) -> list[Path]:
    if not audio_dir.is_dir():
        raise SystemExit(f"Directory not found: {audio_dir}")
    files = sorted(audio_dir.glob("*.wav"))
    if len(files) == 0:
        raise SystemExit(f"No .wav files found in: {audio_dir}")
    if limit > 0:
        files = files[:limit]
    return files


def load_audio_with_torchaudio(path: Path, target_sr: int, audio_len: Optional[float]) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    audio = wav.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    if audio_len is not None:
        expected_len = int(audio_len * target_sr)
        if audio.shape[0] != expected_len:
            raise RuntimeError(
                f"Audio length mismatch ({audio.shape[0] / target_sr:.2f} seconds != {audio_len} seconds).\n\t- {path}"
            )
    return audio


def load_audio_for_model(path: Path, model: ModelLoader, use_custom_loader: bool) -> np.ndarray:
    if use_custom_loader:
        return model.load_wav(path)
    return load_audio_with_torchaudio(path, target_sr=model.sr, audio_len=model.audio_len)


def load_one_file(path: Path, model: ModelLoader, use_custom_loader: bool) -> tuple[Path, np.ndarray]:
    audio = load_audio_for_model(path, model, use_custom_loader)
    return path, audio


def iter_loaded_audio(
    files: list[Path],
    model: ModelLoader,
    workers: int,
    use_custom_loader: bool,
) -> Iterator[tuple[Path, np.ndarray]]:
    if workers <= 1:
        for path in files:
            yield load_one_file(path, model, use_custom_loader)
        return

    max_pending = max(1, workers * 2)
    file_iter = iter(files)
    pending: dict = {}

    def submit(executor: ThreadPoolExecutor) -> bool:
        try:
            path = next(file_iter)
        except StopIteration:
            return False
        future = executor.submit(load_one_file, path, model, use_custom_loader)
        pending[future] = path
        return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(min(max_pending, len(files))):
            submit(executor)

        while pending:
            done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                yield future.result()
                submit(executor)


def compute_embeddings(
    files: list[Path],
    model: ModelLoader,
    workers: int,
    label: str,
) -> np.ndarray:
    # Models overriding load_wav can keep their own shape-specific loading semantics.
    use_custom_loader = model.__class__.load_wav is not ModelLoader.load_wav
    embeddings: list[np.ndarray] = []

    iterator = iter_loaded_audio(files, model=model, workers=workers, use_custom_loader=use_custom_loader)
    with tqdm(total=len(files), desc=f"{label}", unit="file") as pbar:
        for path, audio in iterator:
            with torch.no_grad():
                emb = model.get_embedding(audio)

            if emb.ndim == 1:
                emb = np.expand_dims(emb, axis=0)
            elif emb.ndim != 2:
                raise RuntimeError(
                    f"Expected 1D/2D embedding from model '{model.name}' for '{path}', got shape {emb.shape}."
                )

            embeddings.append(emb)
            pbar.update(1)

    if len(embeddings) == 0:
        raise RuntimeError(f"No embeddings produced for '{label}'.")
    return np.concatenate(embeddings, axis=0)


def compute_scores(gt_emb: np.ndarray, gen_emb: np.ndarray, device: torch.device) -> tuple[float, float]:
    with torch.no_grad():
        fad_score = calc_frechet_distance(
            gt_emb,
            gen_emb,
            cache_dirs=(None, None),
            device=device,
        )
        kad_score = calc_kernel_audio_distance(
            torch.from_numpy(gt_emb),
            torch.from_numpy(gen_emb),
            cache_dirs=(None, None),
            device=device,
            bandwidth=None,
        )

    if isinstance(fad_score, torch.Tensor):
        fad_score = float(fad_score.item())
    else:
        fad_score = float(fad_score)

    if isinstance(kad_score, torch.Tensor):
        kad_score = float(kad_score.item())
    else:
        kad_score = float(kad_score)

    return fad_score, kad_score
