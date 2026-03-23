from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .flamingo import Flamingo
from .flamingo_lm import FlamingoLMMixin
from .utils import extend_instance


def create_model_and_transforms(
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    cache_dir: Optional[str] = None,
    slide_feature_dim: int = 768,
    **flamingo_kwargs,
):
    """
    Build a (patch+optional slide) Flamingo model.

    Changes vs your previous version:
      - Removed cls_type entirely (no cls tokens, no classifier heads).
      - Only adds <image> and <|endofchunk|> special tokens (+ <PAD> if missing).
      - Returns: (model, tokenizer) (same as your current code)
    """

    # Tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
        use_fast=True,
    )

    SPECIAL_TOKENS = ["<image>", "<|endofchunk|>"]
    text_tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # LM
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
        use_safetensors=True,
    )

    # Hack for MPT-1B
    if "mpt-1b-redpajama-200b" in lang_encoder_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings

        extend_instance(lang_encoder, EmbeddingFnMixin)

    # Convert LM to FlamingoLM (adds FlamingoLMMixin methods)
    extend_instance(lang_encoder, FlamingoLMMixin)

    # Find decoder layers attribute name
    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)

    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)

    # Resize embeddings after adding special tokens
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    # Build Flamingo wrapper
    model = Flamingo(
        lang_encoder=lang_encoder,
        eoc_token_id=text_tokenizer.encode("<|endofchunk|>")[-1],
        media_token_id=text_tokenizer.encode("<image>")[-1],
        vis_dim=768,
        tokenizer=text_tokenizer,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        slide_feature_dim=slide_feature_dim,
        **flamingo_kwargs,
    )

    # Freeze all params
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver + gated cross-attn + (optionally) LM input embeddings
    model.perceiver.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Flamingo model initialized with {trainable_params} trainable parameters")
    print(f"Total trainable parameters: {trainable_params/1e6:.2f}M")

    return model, text_tokenizer


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        "Cannot infer decoder layers attribute name for this LM class. "
        "Please pass decoder_layers_attr_name explicitly."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
    "biogptforcausallm": "biogpt.layers",
    "gpt2": "transformer.h",
    "gpt2lmheadmodel": "transformer.h",
}