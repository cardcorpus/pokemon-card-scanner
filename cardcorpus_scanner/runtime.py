"""Running the models through ONNX Runtime on the CPU.

On a CPU, ONNX Runtime runs the same models about twice as fast as PyTorch
(the number reader's small recurrent model most of all), with the same
answers. Each model is exported from its PyTorch weights the first time it
is needed and kept in a cache folder, keyed by a hash of its weights, so a
new release's weights get new files. Every export is checked against
PyTorch on a test input before it is used; one that doesn't agree is left
out and that model stays on PyTorch, with a warning.

The export needs the `onnx` package (a dependency); the run needs
`onnxruntime` (one too, for the language model).
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

# bump when an export changes (opset, input shapes): old cache files are then left alone
EXPORT_VERSION = 1
OPSET = 17


def cache_folder():
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "cardcorpus-scanner" / "onnx"


def weights_digest(module):
    """A short hash of a module's weights (parameters and buffers, in order)."""
    h = hashlib.sha256()
    for name, t in list(module.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


class OnnxModel:
    """A model in ONNX Runtime that is called like the PyTorch module it was
    exported from: tensors in, tensors (or a tuple of them) out, on the CPU."""

    def __init__(self, path, threads=None, outputs=1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.outputs = outputs
        self.path = Path(path)

    def __call__(self, x):
        import torch

        outs = self.session.run(None, {"x": x.detach().cpu().float().numpy()})
        outs = [torch.from_numpy(o) for o in outs]
        return tuple(outs) if self.outputs > 1 else outs[0]

    def eval(self):
        return self

    def to(self, *_args, **_kw):
        return self


def _agree(ref, got):
    """Outputs within float32 noise of each other: cosine of every row over 0.9999."""
    for r, g in zip(ref, got):
        r = r.detach().float().reshape(r.shape[0], -1).numpy()
        g = g.detach().float().reshape(g.shape[0], -1).numpy()
        if r.shape != g.shape:
            return False
        num = (r * g).sum(1)
        den = np.linalg.norm(r, axis=1) * np.linalg.norm(g, axis=1) + 1e-12
        if (num / den).min() < 0.9999:
            return False
    return True


def to_onnx(module, name, example, dynamic, threads=None, folder=None, log=None):
    """The module as an OnnxModel, exported once into the cache (folder),
    or None when it can't be exported or its export doesn't match PyTorch.

    example: an input tensor of a shape the scanner uses; dynamic: the input's
    axes that vary ({0: "batch"}, ...)."""
    import torch

    log = log or (lambda msg: print(msg, file=sys.stderr))
    module = module.eval()
    with torch.no_grad():
        ref = module(example)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    folder = Path(folder) if folder else cache_folder()
    path = folder / f"{name}-{weights_digest(module)}-v{EXPORT_VERSION}.onnx"
    try:
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            outs = [f"out{i}" for i in range(len(ref))]
            part = path.with_name(path.name + f".{os.getpid()}.part")
            # the TorchScript exporter: dynamo=False from torch 2.5 on, where the default became the new one
            import inspect

            legacy = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
            with torch.no_grad():
                torch.onnx.export(module, (example,), str(part), input_names=["x"], output_names=outs,
                                  dynamic_axes={"x": dynamic, **{o: {0: "batch"} for o in outs}},
                                  opset_version=OPSET, **legacy)
            os.replace(part, path)
        model = OnnxModel(path, threads, outputs=len(ref))
        # the example's batch and a smaller one (the scanner's last batch is often short)
        checks = [(example, ref)]
        if example.shape[0] > 1:
            small = example[:3].clone()
            with torch.no_grad():
                out = module(small)
            checks.append((small, out if isinstance(out, (tuple, list)) else (out,)))
        for x, want in checks:
            got = model(x)
            if not _agree(want, got if isinstance(got, tuple) else (got,)):
                log(f"ONNX Runtime: the {name}'s export doesn't match PyTorch; it stays on PyTorch")
                return None
        return model
    except Exception as e:  # an exporter or runtime that can't take this model: stay on PyTorch
        log(f"ONNX Runtime: couldn't use the {name} ({type(e).__name__}: {e}); it stays on PyTorch")
        return None


def available():
    try:
        import onnx  # noqa: F401  (torch.onnx.export needs it)
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True
