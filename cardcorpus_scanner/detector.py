"""Stage 1: find the cards in a photo.

QuadDet is a small corner-keypoint detector: a heatmap of card centres plus,
at every cell, the card's four corners. Each card comes out as a
perspective quad, not a rotated rectangle.

    image (3, H, W), RGB in [0, 1]
      -> MobileNetV4 features at strides 4, 8, 16, 32
      -> top-down FPN to stride 4, with a pooled context block on stride 32
      -> per cell (H/4 x W/4):
           heat    centre-ness logit
           offset  where the centre lies inside the cell, in cells
           scale   log of the card's size, sqrt(quad area) / stride
           shape   8 values: the corners minus the centre, over the size

Which corner comes first is left to stage 2, which turns each card upright.

The photo goes in with its long side resized to 1280
(bilinear, `bilinear` below), placed at the top left of the smallest of the
canvases 960x1280, 1280x960 and 1280x1280 that holds it, grey (114)
elsewhere.
"""
import numpy as np

STRIDE = 4
CANVASES = [(960, 1280), (1280, 960), (1280, 1280)]  # (width, height)
SIDE = 1280
CONTAIN = 0.75


def _modules():
    """The model's layers, built on first use so that importing this module
    doesn't need torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvBN(nn.Sequential):
        def __init__(self, cin, cout, k=1, groups=1, act=True):
            super().__init__(
                nn.Conv2d(cin, cout, k, padding=k // 2, groups=groups, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True) if act else nn.Identity(),
            )

    class SepBlock(nn.Module):
        """Depthwise k x k then pointwise, with a residual."""

        def __init__(self, c, k=3):
            super().__init__()
            self.dw = ConvBN(c, c, k, groups=c, act=False)
            self.pw = ConvBN(c, c, 1)

        def forward(self, x):
            return x + self.pw(self.dw(x))

    class Context(nn.Module):
        """Max pools of growing reach on the coarsest map, so a card filling
        the photo still sees its own edges from its centre."""

        def __init__(self, c):
            super().__init__()
            self.pool = nn.MaxPool2d(5, 1, 2)
            self.mix = ConvBN(4 * c, c)

        def forward(self, x):
            a = self.pool(x)
            b = self.pool(a)
            return self.mix(torch.cat([x, a, b, self.pool(b)], 1))

    class QuadDet(nn.Module):
        def __init__(self, backbone="mobilenetv4_conv_small", width=64, head_blocks=2, corners=False, **_):
            super().__init__()
            import timm

            self.backbone = timm.create_model(backbone, features_only=True, out_indices=(1, 2, 3, 4),
                                              pretrained=False)
            chans = self.backbone.feature_info.channels()
            self.lateral = nn.ModuleList(ConvBN(c, width) for c in chans)
            self.context = Context(width)
            self.smooth = nn.ModuleList(SepBlock(width, 5) for _ in chans[:-1])
            self.head = nn.Sequential(*[SepBlock(width, 5) for _ in range(head_blocks)])
            self.heat = nn.Conv2d(width, 1, 1)
            self.reg = nn.Conv2d(width, 2 + 1 + 8, 1)
            self.corners = corners
            if corners:
                self.corner = nn.Conv2d(width, 3, 1)

        def forward(self, x):
            feats = self.backbone(x)
            p = self.context(self.lateral[-1](feats[-1]))
            for i in range(len(feats) - 2, -1, -1):
                p = self.smooth[i](self.lateral[i](feats[i]) + F.interpolate(p, scale_factor=2.0, mode="nearest"))
            y = self.head(p)
            if self.corners:
                return self.heat(y), self.reg(y), self.corner(y)
            return self.heat(y), self.reg(y)

    return QuadDet


def build(config):
    """QuadDet with no weights, from the `detector.model` settings."""
    return _modules()(**(config or {}))


def export_tensor(heat, reg, corner=None):
    """(B, 12, h, w): per cell its peak score (zero unless the cell is its 3 x
    3 neighbourhood's maximum) and the 11 raw regression values; with the
    corner head, 15 channels. The ONNX export ends here; `decode` makes quads
    of it."""
    import torch
    import torch.nn.functional as F

    score = torch.sigmoid(heat)
    peak = (F.max_pool2d(score, 3, 1, 1) == score).to(score.dtype)
    if corner is None:
        return torch.cat([score * peak, reg], 1)
    cscore = torch.sigmoid(corner[:, :1])
    cpeak = (F.max_pool2d(cscore, 3, 1, 1) == cscore).to(cscore.dtype)
    return torch.cat([score * peak, reg, cscore * cpeak, corner[:, 1:]], 1)


def decode(out, conf=0.25, top=300, stride=STRIDE, corner_conf=0.1, snap=0.08):
    """(12 or 15, h, w) from export_tensor -> [(score, quad 4 x 2 in canvas
    pixels)], best first, ties by position. With the corner head, each
    corner moves to the most confident corner peak within snap x the card's
    size of it, if there is one."""
    out = np.asarray(out, dtype=np.float32)
    score = out[0]
    peaks = None
    if out.shape[0] == 15:
        cy_, cx_ = np.nonzero(out[12] > corner_conf)
        peaks = np.stack([(cx_ + 0.5 + out[13, cy_, cx_].astype(np.float64)) * stride,
                          (cy_ + 0.5 + out[14, cy_, cx_].astype(np.float64)) * stride,
                          out[12, cy_, cx_].astype(np.float64)], 1)
    ys, xs = np.nonzero(score > conf)
    order = sorted(range(len(ys)), key=lambda i: (-float(score[ys[i], xs[i]]), int(ys[i]), int(xs[i])))[:top]
    found = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        r = out[1:, y, x].astype(np.float64)
        cx = (x + 0.5 + r[0]) * stride
        cy = (y + 0.5 + r[1]) * stride
        size = float(np.exp(np.clip(r[2], -4.0, 8.0))) * stride
        quad = [[cx + r[3 + 2 * k] * size, cy + r[4 + 2 * k] * size] for k in range(4)]
        if peaks is not None and len(peaks):
            for k in range(4):
                d = np.hypot(peaks[:, 0] - quad[k][0], peaks[:, 1] - quad[k][1])
                near = np.nonzero(d <= snap * size)[0]
                if len(near):
                    j = near[np.argmax(peaks[near, 2])]
                    quad[k] = [float(peaks[j, 0]), float(peaks[j, 1])]
        found.append((float(score[y, x]), quad))
    return found


def bilinear(img, width, height):
    """img (h, w, 3) uint8 resized to width x height by OpenCV's INTER_LINEAR
    rule (no smoothing when shrinking), interpolated in 32-bit floats and
    rounded half away from zero. Spelled out so a port can compute the same
    bytes."""
    def taps(count, source):
        pos = (np.arange(count, dtype=np.float64) + 0.5) * (source / count) - 0.5
        below = np.floor(pos).astype(np.int64)
        weight = (pos - below).astype(np.float32)
        low = below < 0
        high = below >= source - 1
        below[low], weight[low] = 0, 0
        below[high], weight[high] = source - 1, 0
        return below, np.minimum(below + 1, source - 1), weight
    x0, x1, wx = taps(width, img.shape[1])
    y0, y1, wy = taps(height, img.shape[0])
    src = img.astype(np.float32)
    top, bottom = src[y0], src[y1]
    wx = wx[None, :, None]
    upper = top[:, x0] + (top[:, x1] - top[:, x0]) * wx
    lower = bottom[:, x0] + (bottom[:, x1] - bottom[:, x0]) * wx
    out = upper + (lower - upper) * wy[:, None, None]
    return np.floor(out.astype(np.float64) + 0.5).astype(np.uint8)


def fit_canvas(img, side=SIDE, canvases=CANVASES):
    """(canvas uint8 HxWx3, (x gain, y gain)): the photo shrunk so its long
    side is `side`, at the canvas's top left."""
    h, w = img.shape[:2]
    gain = side / max(h, w)
    nw, nh = round(w * gain), round(h * gain)
    fitting = [c for c in canvases if c[0] >= nw and c[1] >= nh]
    cw, ch = min(fitting, key=lambda c: c[0] * c[1]) if fitting else max(canvases, key=lambda c: c[0] * c[1])
    canvas = np.full((ch, cw, 3), 114, np.uint8)
    canvas[:nh, :nw] = bilinear(img, nw, nh)
    return canvas, (nw / w, nh / h)


def containment(dets, width, height, threshold=CONTAIN, resolution=400):
    """Drop a box covered >= threshold by a kept, more confident one (an
    artwork window boxed inside its card): areas on a raster `resolution` px
    on the photo's long side. dets: [(score, quad)]."""
    if len(dets) <= 1:
        return dets
    from PIL import Image, ImageDraw

    scale = max(width, height) / resolution
    masks = []
    for _, quad in dets:
        m = Image.new("1", (int(width / scale), int(height / scale)), 0)
        ImageDraw.Draw(m).polygon([(x / scale, y / scale) for x, y in quad], fill=1)
        masks.append(np.array(m, dtype=bool))
    order = sorted(range(len(dets)), key=lambda i: -dets[i][0])
    keep = []
    for i in order:
        area = masks[i].sum()
        if area and not any((masks[i] & masks[j]).sum() / area >= threshold for j in keep):
            keep.append(i)
    return [dets[i] for i in sorted(keep)]


class Detector:
    """QuadDet with its weights: photo (h, w, 3 uint8 RGB) -> [(score, quad in
    photo pixels)], before containment."""

    def __init__(self, weights, config, conf, device="cpu"):
        """weights: a safetensors file, or a state dict already in memory."""
        import torch

        from .tower import state_dict

        self.torch = torch
        self.model = build(config)
        self.model.load_state_dict(state_dict(weights))
        self.model.to(device).eval()
        self.device = device
        self.conf = conf

    def __call__(self, img):
        canvas, (gx, gy) = fit_canvas(img)
        x = self.torch.from_numpy(canvas).permute(2, 0, 1)[None].float().div(255).to(self.device)
        with self.torch.no_grad():
            out = export_tensor(*self.model(x))[0].float().cpu().numpy()
        return [(s, [[px / gx, py / gy] for px, py in q]) for s, q in decode(out, self.conf)]

