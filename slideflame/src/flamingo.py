# src/flamingo.py
import torch
from torch import nn
from .helpers import PerceiverResampler


class Flamingo(nn.Module):
    def __init__(
        self,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        tokenizer,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
        slide_feature_dim: int = 768,
    ):
        """
        Patch+Slide (optional) Flamingo-style model for report generation.

        Changes vs your current version:
          - Removed classifier heads (organ/diagnosis) entirely.
          - PerceiverResampler now supports slide_query=None (patch-only).
          - Forward returns a dict-like HF output (unchanged behavior),
            but no longer computes / returns cls logits.
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.vis_dim = vis_dim
        self.tokenizer = tokenizer

        self.lang_encoder = lang_encoder
        self.config = self.lang_encoder.config
        self.generation_config = self.lang_encoder.generation_config

        # Determine language hidden size robustly
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model
        else:
            self.lang_dim = lang_encoder.get_input_embeddings().embedding_dim

        # Perceiver: slide_feature_dim is only used if slide features are provided
        self.perceiver = PerceiverResampler(dim=self.vis_dim, slide_dim=slide_feature_dim)

        # Init Flamingo cross-attn layers inside the LM
        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            lang_hidden_size=self.lang_dim,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            gradient_checkpointing=gradient_checkpointing,
        )

        self._use_gradient_checkpointing = gradient_checkpointing
        self.perceiver._use_gradient_checkpointing = gradient_checkpointing
        self.last_vision_x = None
        self._save_last_vision_x = False

    # ---- HF / PEFT generation compatibility shims ----
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.lang_encoder.prepare_inputs_for_generation(*args, **kwargs)

    def get_input_embeddings(self, *args, **kwargs):
        return self.lang_encoder.get_input_embeddings(*args, **kwargs)

    def get_output_embeddings(self, *args, **kwargs):
        return self.lang_encoder.get_output_embeddings(*args, **kwargs)

    # -------------------------------------------------

    def forward(
        self,
        patch_features: torch.Tensor,
        slide_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        save_perceiver_attn: bool = False,
        save_last_vision_x: bool = False,
        **kwargs,
    ):
        """
        Args:
            patch_features: (B, T_img, N, D)  where D==vis_dim
            slide_features: (B, D_slide) or None (patch-only)
            input_ids:      (B, T_txt)
            attention_mask: (B, T_txt)
            labels:         (B, T_txt) with -100 masking
        """
        assert self.lang_encoder.initialized_flamingo, (
            "Flamingo layers are not initialized. Please call `init_flamingo` first."
        )

        self._save_last_vision_x = bool(save_last_vision_x)
        self.last_vision_x = None

        self._encode_vision_x(
            patch_features=patch_features,
            slide_features=slide_features,
            save_perceiver_attn=save_perceiver_attn, # can be None
        )
        self._condition_media_locations(input_ids=input_ids)

        output = self.lang_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    @torch.no_grad()
    def generate(
        self,
        patch_features: torch.Tensor,
        slide_features: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Generate text conditioned on patch features and optional slide features.

        Args:
            patch_features: (B, T_img, N, D)
            slide_features: (B, D_slide) or None
            lang_x: (B, T_txt)
        """
        kwargs.setdefault("return_dict_in_generate", True)
        kwargs.setdefault("output_scores", True)
        kwargs.setdefault("num_beams", 1)

        num_beams = int(kwargs["num_beams"])
        do_sample = bool(kwargs.get("do_sample", False))
        num_return_sequences = int(kwargs.get("num_return_sequences", 1))

        if num_beams > 1:
            expand_size = num_beams
        elif do_sample and num_return_sequences > 1:
            expand_size = num_return_sequences
        else:
            expand_size = 1

        if expand_size > 1:
            patch_features = patch_features.repeat_interleave(expand_size, dim=0)
            if slide_features is not None:
                slide_features = slide_features.repeat_interleave(expand_size, dim=0)

        # used by FlamingoLM caching logic
        self.lang_encoder.cached_input_ids = lang_x
        self.lang_encoder._use_cached_vision_x = True

        self._encode_vision_x(
            patch_features=patch_features,
            slide_features=slide_features,
        )

        eos_token_id = kwargs.pop("eos_token_id", self.eoc_token_id)

        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
        return output

    def _encode_vision_x(self, patch_features: torch.Tensor, slide_features: torch.Tensor = None, save_perceiver_attn: bool = False):
        """
        Convert patch features into Perceiver latents and condition all decoder layers.

        patch_features expected shape: (B, T_img, N, D)
        We reshape to (B, T_img, F=1, v=N, D) for PerceiverResampler.
        slide_features can be None (patch-only).
        """
        if patch_features.ndim != 4:
            raise ValueError(f"Expected patch_features shape [B,T,N,D], got {patch_features.shape}")
        

        B, T_img, V, D = patch_features.shape
        patch_features = patch_features.view(B, T_img, 1, V, D)

        # PerceiverResampler supports slide_query=None now
        vision_x = self.perceiver(x=patch_features, slide_query=slide_features, save_attn=save_perceiver_attn)
        if self._save_last_vision_x:
            self.last_vision_x = vision_x
            self.last_vision_x.retain_grad()
        # if not hasattr(self, "_dbg_hook_once"):
        #     self._dbg_hook_once = True
        #     def _hook(g):
        #         print("[DEBUG] vision_x GOT GRAD. norm =", float(g.norm()))
        #     vision_x.register_hook(_hook)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

    def _condition_media_locations(self, input_ids: torch.Tensor):
        """
        Condition decoder layers on positions of <image> token.
        """
        media_locations = input_ids == self.media_token_id
        #print(f"Media locations shape: {media_locations}")
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_media_locations(media_locations)

    def cache_media(self, patch_features: torch.Tensor, slide_features: torch.Tensor = None):
        """
        Pre-process and cache the visual features for rapid generation.
        """
        self._encode_vision_x(patch_features=patch_features, slide_features=slide_features)
        self.lang_encoder._use_cached_vision_x = True

    def uncache_media(self):
        """
        Clear all conditioning.
        """
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False