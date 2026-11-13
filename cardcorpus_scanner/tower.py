"""The card matcher's and the print picker's image towers.

Each is a timm backbone, optionally followed by a linear projection (no
bias) to the index's width, described by a spec in pipeline.json:

  arch         the timm model name
  size         the square the tower takes, in pixels ([h, w] for the
               picker's whole card)
  interpolation, resize_mode
               "bilinear" + "shortest": the shorter side resized to `size`
               (antialiased), then the centre square
  mean, std    per-channel normalisation of RGB in [0, 1]
  head         "keep" keeps the timm model's own head (PE-Core's projection
               into its CLIP space); otherwise the pooled features are the
               output
  projection   null, or the backbone's output width: a linear map from it to
               `dim`
  dim          the embedding's width (512)
  logit_limit  null, or the largest attention logit the tower computes
               (`clamped_attention`): what 16-bit arithmetic can hold, so
               that PyTorch, ONNX and Core ML agree
"""
import math
from contextlib import contextmanager


@contextmanager
def clamped_attention(limit=64000.0):
    """Attention as timm computes it, except that q.k is computed on q and k
    each scaled down by 16 and clamped to +-limit/256 before it is scaled
    back. In 32 bits this is the original for every logit within +-limit; in
    16 bits it keeps the product from overflowing. DINOv3's and PE-Core's
    logits can reach 10^6 (a register or class token every query attends
    to)."""
    import torch.nn.functional as F

    original = F.scaled_dot_product_attention

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        assert attn_mask is None and not is_causal and dropout_p == 0.0
        s = scale if scale is not None else 1 / math.sqrt(q.shape[-1])
        logits = (q * (s / 16)) @ (k / 16).transpose(-2, -1)
        logits = logits.clamp(-limit / 256, limit / 256) * 256
        return logits.softmax(dim=-1) @ v

    F.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


def hw(spec):
    """(height, width) of the tower's input."""
    size = spec["size"]
    return (size, size) if isinstance(size, int) else tuple(size)


def build(spec):
    """The tower for a spec, without weights. Its output is unnormalised."""
    import timm
    import torch.nn as nn

    class Tower(nn.Module):
        def __init__(self, backbone, proj=None, logit_limit=None):
            super().__init__()
            self.backbone = backbone
            self.proj = proj
            self.logit_limit = logit_limit

        def forward(self, x):
            if self.logit_limit:
                with clamped_attention(self.logit_limit):
                    f = self.backbone(x)
            else:
                f = self.backbone(x)
            return self.proj(f) if self.proj is not None else f

    kwargs = {} if spec.get("head") == "keep" else {"num_classes": 0}
    try:
        # ViT-like models take img_size (their position embedding's grid)
        backbone = timm.create_model(spec["arch"], pretrained=False, img_size=hw(spec), **kwargs)
    except TypeError:  # convnets have no such argument
        backbone = timm.create_model(spec["arch"], pretrained=False, **kwargs)
    width = spec.get("projection")
    proj = nn.Linear(width, spec["dim"], bias=False) if width is not None else None
    return Tower(backbone, proj, spec.get("logit_limit"))


def load(weights, spec, device="cpu"):
    """The tower with its weights (a safetensors file, or a state dict
    already in memory), in eval mode."""
    tower = build(spec)
    tower.load_state_dict(state_dict(weights))
    return tower.eval().to(device)


def state_dict(weights):
    """A state dict from a safetensors file, or one given as it is."""
    if isinstance(weights, dict):
        return weights
    from safetensors.torch import load_file

    return load_file(str(weights))


def normalise(spec):
    import torchvision.transforms as T

    return T.Normalize(mean=tuple(spec["mean"]), std=tuple(spec["std"]))


def preprocess(spec):
    """PIL image -> normalised tensor for the matcher: the shorter side
    resized to `size` (antialiased), the centre square, RGB in [0, 1],
    mean/std."""
    import torchvision.transforms as T

    size = spec["size"]
    interp = T.InterpolationMode.BILINEAR if spec.get("interpolation") == "bilinear" else T.InterpolationMode.BICUBIC
    mode = spec.get("resize_mode") or "shortest"
    if mode == "shortest":
        if not isinstance(size, int):
            raise ValueError("resize_mode 'shortest' takes a square size")
        steps = [T.Resize(size, interpolation=interp), T.CenterCrop((size, size))]
    elif mode == "squash":
        steps = [T.Resize(hw(spec), interpolation=interp)]
    else:
        raise ValueError(f"resize_mode {mode!r}: 'shortest' or 'squash'")
    return T.Compose(steps + [lambda im: im.convert("RGB") if im.mode != "RGB" else im, T.ToTensor(), normalise(spec)])


def picker_transform(spec):
    """PIL image -> normalised tensor for the print picker: the whole card
    resized to (h, w), bicubic."""
    import torchvision.transforms as T

    return T.Compose([T.Resize(hw(spec), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), normalise(spec)])
