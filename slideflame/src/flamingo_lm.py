import torch
import torch.nn as nn
from .helpers import GatedCrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive


class FlamingoLayer(nn.Module):
    """
    Wrapper around an optional GatedCrossAttentionBlock + the original decoder layer.
    """

    def __init__(self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer

        self.vis_x = None
        self.media_locations = None
        self.use_cached_media = False

        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = gradient_checkpointing
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def is_conditioned(self) -> bool:
        return self.vis_x is not None and self.media_locations is not None

    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        self.media_locations = media_locations

    def condition_use_cached_media(self, use_cached_media: bool):
        self.use_cached_media = bool(use_cached_media)

    def forward(self, lang_x, *args, **kwargs):
        # Cross attention (optional)
        if self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")
            if self.media_locations is None:
                raise ValueError("media_locations must be conditioned before forward pass")

            lang_x = self.gated_cross_attn_layer(
                lang_x,
                self.vis_x,
                media_locations=self.media_locations,
                use_cached_media=self.use_cached_media,
            )

        # Normal decoder layer
        lang_x = self.decoder_layer(lang_x, *args, **kwargs)
        return lang_x


class FlamingoLMMixin(nn.Module):
    """
    Mixin to add Flamingo cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(
        self,
        media_token_id,
        lang_hidden_size,
        vis_hidden_size,
        cross_attn_every_n_layers,
        gradient_checkpointing,
    ):
        """
        Insert gated cross-attention layers into the decoder stack.
        """
        self.old_decoder_blocks = self._get_decoder_layers()
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(dim=lang_hidden_size, dim_visual=vis_hidden_size)
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )

        self.init_flamingo_layers(gradient_checkpointing)
        self.media_token_id = media_token_id
        self.initialized_flamingo = True

        # generation caching flag
        self._use_cached_vision_x = False

    def init_flamingo_layers(self, gradient_checkpointing):
        """
        Wrap each decoder block with FlamingoLayer.
        """
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    FlamingoLayer(gated_cross_attn_layer, decoder_layer, gradient_checkpointing)
                    for gated_cross_attn_layer, decoder_layer in zip(
                        self.gated_cross_attn_layers, self.old_decoder_blocks
                    )
                ]
            )
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Condition Flamingo layers on media locations and cached-media status,
        then call the base model forward.
        """
        if not getattr(self, "initialized_flamingo", False):
            raise ValueError("Flamingo layers are not initialized. Please call `init_flamingo` first.")

        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape, dtype=torch.long)
        else:
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.to(dtype=torch.long)

        media_locations = input_ids == self.media_token_id

        use_cached_media_locations = (
            self._use_cached_vision_x
            and self.is_conditioned()
            and not media_locations.any()
        )

        for layer in self._get_decoder_layers():
            if not use_cached_media_locations:
                layer.condition_media_locations(media_locations)
            layer.condition_use_cached_media(use_cached_media_locations)

        # Do NOT force output_hidden_states=True anymore.
        # If you ever need it later, pass output_hidden_states=True from the caller explicitly.

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        return super().forward(**kwargs)

    def is_conditioned(self) -> bool:
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_use_cached_media(False)