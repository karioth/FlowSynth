"""
Unified text encoder using CLAP and T5 models.

Provides standardized output keys (CPU pool naming convention).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, ClapModel, ClapProcessor, T5EncoderModel


class TextEncoder:
    """
    Unified text encoder using CLAP and T5 models.

    Loads models once and provides single-caption encoding.
    Output payload uses standardized keys:
        - clap_embedding: [512] normalized embedding
        - clap_last_hidden: [Lc, 768] hidden states (no padding)
        - clap_len: int
        - t5_last_hidden: [Lt, 1024] hidden states (no padding)
        - t5_len: int
    """

    def __init__(
        self,
        clap_model_name: str = "laion/larger_clap_music",
        t5_model_name: str = "google/flan-t5-large",
        device: str = "cpu",
    ):
        """
        Initialize text encoder with CLAP and T5 models.

        Args:
            clap_model_name: HuggingFace model ID for CLAP
            t5_model_name: HuggingFace model ID for T5 encoder
            device: Device to load models on
        """
        self.device = device

        # Load CLAP
        self.clap_model = (
            ClapModel.from_pretrained(clap_model_name, use_safetensors=True)
            .eval()
            .to(device)
        )
        self.clap_processor = ClapProcessor.from_pretrained(clap_model_name)

        # Load T5
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        self.t5_encoder = (
            T5EncoderModel.from_pretrained(t5_model_name, use_safetensors=True)
            .eval()
            .to(device)
        )

    @torch.no_grad()
    def encode(self, caption: str) -> dict:
        """
        Encode a single caption.

        Args:
            caption: Text caption to encode

        Returns:
            Dictionary with standardized keys (see class docstring)
        """
        # CLAP encoding
        clap_embed, clap_last_hidden, clap_len = self._encode_clap(caption)

        # T5 encoding
        t5_last_hidden, t5_len = self._encode_t5(caption)

        return {
            "clap_embedding": clap_embed.cpu().to(torch.float32),
            "clap_last_hidden": clap_last_hidden.cpu().to(torch.float32),
            "clap_len": int(clap_len),
            "t5_last_hidden": t5_last_hidden.cpu().to(torch.float32),
            "t5_len": int(t5_len),
        }

    def _encode_clap(self, caption: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Internal CLAP encoding.

        Returns:
            (embedding, last_hidden, length)
        """
        inputs = self.clap_processor(
            text=[caption],
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        text_kwargs = {"input_ids": inputs["input_ids"]}
        if "token_type_ids" in inputs:
            text_kwargs["token_type_ids"] = inputs["token_type_ids"]

        text_out = self.clap_model.text_model(
            return_dict=True, output_hidden_states=False, **text_kwargs
        )

        proj = (
            self.clap_model.text_projection
            if hasattr(self.clap_model, "text_projection")
            else self.clap_model.text_model.text_projection
        )
        embeds_unnorm = proj(text_out.pooler_output)
        embeds = F.normalize(embeds_unnorm, dim=-1)

        last_hidden = text_out.last_hidden_state[0]  # [Lc, 768]
        return embeds[0], last_hidden, int(last_hidden.shape[0])

    def _encode_t5(self, caption: str) -> tuple[torch.Tensor, int]:
        """
        Internal T5 encoding.

        Returns:
            (last_hidden, length)
        """
        t5_inputs = self.t5_tokenizer(
            [caption],
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        t5_inputs = {k: v.to(self.device) for k, v in t5_inputs.items()}

        t5_out = self.t5_encoder(input_ids=t5_inputs["input_ids"], return_dict=True)
        t5_last = t5_out.last_hidden_state[0]  # [Lt, 1024]

        return t5_last, int(t5_last.shape[0])
