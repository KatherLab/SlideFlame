"""
Based on: https://github.com/lucidrains/flamingo-pytorch
Patch-only support added: PerceiverResampler can run with slide_query=None.
"""

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn
import torch.nn.functional as F


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.last_attn = None
        self.last_media_len = None

    def forward(self, x, latents, save_attn: bool = False):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latents (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if save_attn:
        # attn: (b, h, T, n_lat, n_media + n_lat)
            self.last_attn = attn
            self.last_attn.retain_grad()     # required to read attn.grad later
            self.last_media_len = x.shape[-2]  # n_media (after flatten)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=512,
        ff_mult=4,
        slide_dim=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.heads = heads
        self.num_latents = num_latents
        self.frame_embs = None
        self.media_time_embs = None
        self.ff_mult = ff_mult
        slide_dim = slide_dim if slide_dim is not None else dim

        # slide projection stays for backward compatibility even if patch_only is used
        self.slide_proj = nn.Linear(slide_dim, dim) if slide_dim != dim else nn.Identity()

        # Learnable latents always exist (patch-only uses these only)
        self.learnable_latents = nn.Parameter(torch.randn(num_latents, dim))

        # Slide gate (kept for checkpoint compatibility; unused if slide_query=None)
        self.slide_gate = nn.Parameter(torch.tensor(0.1))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, slide_query=None, save_attn: bool = False):
        """
        Args:
            x (torch.Tensor): Patch-level features.
                Shape (b, T, F, v, D)
            slide_query (torch.Tensor or None): Slide-level features.
                Shape (b, slide_dim) or None (patch-only mode)

        Returns:
            latents: (b, T, num_latents, D)
        """
        b, T, F, v = x.shape[:4]

        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs

        # flatten frames*patches
        x = rearrange(x, "b T F v d -> b T (F v) d")

        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # Always start from learnable latents
        learned_latents = self.learnable_latents.unsqueeze(0).unsqueeze(0).expand(b, T, -1, -1)

        # If slide_query is provided, inject slide-conditioned latents (your original behavior)
        if slide_query is not None:
            projected_query = self.slide_proj(slide_query)          # (b, dim)
            projected_query = rearrange(projected_query, "b d -> b 1 d")  # (b, 1, dim)
            slide_latents_template = projected_query.unsqueeze(2)   # (b, 1, 1, dim)
            slide_latents = slide_latents_template.expand(-1, T, self.num_latents, -1)  # (b, T, n, dim)
            latents = learned_latents + self.slide_gate * slide_latents
        else:
            # Patch-only mode: no slide injection at all
            latents = learned_latents

        for attn, ff in self.layers:
            latents = attn(x, latents, save_attn=save_attn) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.only_attend_immediate_media = only_attend_immediate_media

        self.last_attn = None

    def forward(self, x, media, media_locations=None, use_cached_media=False, save_attn: bool = False):
        save_attn = save_attn or getattr(self, "_save_attn", False)
        if not use_cached_media:
            assert media_locations.shape[1] == x.shape[1], (
                f"media_location.shape is {media_locations.shape} but x.shape is {x.shape}"
            )

        T_txt = x.shape[1]
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)

        if exists(media_locations):
            media_time = torch.arange(T_img, device=x.device) + 1

            if use_cached_media:
                text_time = torch.full(
                    (x.shape[0], T_txt), fill_value=T_img, device=x.device, dtype=torch.long
                )
            else:
                text_time = media_locations.cumsum(dim=-1)

            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        if save_attn:
            self.last_attn = attn
            self.last_attn.retain_grad()

        if exists(media_locations) and self.only_attend_immediate_media:
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, "b i -> b 1 i 1")
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.55]))
        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, media, media_locations=None, use_cached_media=False):
        x = self.attn(
            x,
            media,
            media_locations=media_locations,
            use_cached_media=use_cached_media,
        ) * self.attn_gate.tanh() + x

        x = self.ff(x) * self.ff_gate.tanh() + x
        return x