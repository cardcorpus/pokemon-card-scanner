"""A weights folder with random models and a small made-up index, in the
release's layout, for tests that check the scanner runs, not what it says."""
import gzip
import json
from pathlib import Path

import numpy as np

CHARSET = "0123456789/No.-ABCDEFGHIJKLMOPQRSTUVWXYZ"
MATCHER = {"arch": "vit_small_patch16_dinov3.lvd1689m", "size": 224, "interpolation": "bilinear",
           "resize_mode": "shortest", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
           "head": None, "projection": 384, "dim": 512, "logit_limit": 64000.0}
PICKER = {"arch": "vit_pe_core_small_patch16_384.fb", "interpolation": "bilinear", "resize_mode": "shortest",
          "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "head": "keep", "projection": None, "dim": 512,
          "logit_limit": 64000.0}
DETECTOR = {"backbone": "mobilenetv4_conv_small", "width": 64, "head_blocks": 2, "corners": True}
SLOTS = [{"cx": 0.85, "cy": 0.955, "w": 0.12, "h": 0.022}, {"cx": 0.15, "cy": 0.955, "w": 0.12, "h": 0.022}]


def make(folder, cards=24, seed=0):
    import torch
    from safetensors.torch import save_file

    from cardcorpus_scanner import detector, number_reader, tower

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    W = Path(folder)
    (W / "models").mkdir(parents=True, exist_ok=True)
    (W / "index").mkdir(exist_ok=True)
    cfg = {"version": "test", "detector": {"conf": 0.2, "contain": 0.75, "model": DETECTOR},
           "embedder": MATCHER, "picker": {"size": [448, 320], "backbone": PICKER}}
    (W / "pipeline.json").write_text(json.dumps(cfg))

    def save(module, name):
        save_file({k: v.contiguous() for k, v in module.state_dict().items()}, str(W / "models" / name))

    save(detector.build(DETECTOR), "detector.safetensors")
    save(tower.build(MATCHER), "matcher.safetensors")
    save(tower.build({**MATCHER, **PICKER, "size": [448, 320]}), "picker.safetensors")
    save(number_reader.build(CHARSET), "number-reader.safetensors")
    (W / "models" / "number-reader.json").write_text(json.dumps(
        {"charset": CHARSET, "strip": [48, 384], "slots": SLOTS, "denominators": [64, 102]}))
    save_file({"orient_probe": torch.randn(4, 257), "orient_hidden": torch.randn(256, 513) * 0.1,
               "back_prototype": torch.nn.functional.normalize(torch.randn(512), dim=0)},
              str(W / "models" / "heads.safetensors"))

    rows, handles = [], []
    for i in range(cards):
        h = f"{i:032x}"
        handles.append(h)
        rows.append({"handle": h, "kind": "card", "set": "international/test-set", "card": str(i + 1),
                     "name": f"Test Card {i % 8}", "set_name": "Test Set", "number": f"{i + 1}/64",
                     "family": "international" if i % 3 else "japanese", "language": "en" if i % 3 else "ja",
                     "release": f"r{i % 2}", "artwork_group": f"a{i % 8}", "release_date": f"2000-01-{i % 28 + 1:02d}",
                     "rarity": "Rare Holo" if i % 4 == 0 else "Common", "picture": h})
    rows.append({"handle": "view-1", "kind": "view", "name": "Test Card 0", "family": "international"})
    handles.append("view-1")
    with gzip.open(W / "index" / "cards.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    e = rng.standard_normal((len(handles), 512)).astype(np.float32)
    np.save(W / "index" / "embeddings.npy", (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float16))
    (W / "index" / "rows.json").write_text(json.dumps(handles))
    (W / "index" / "dupe-groups.json").write_text(json.dumps({"view-1": "g1", handles[0]: "g1"}))
    prints = [f"international/test-set/{i + 1}" for i in range(8)]
    p = rng.standard_normal((len(prints), 512)).astype(np.float32)
    np.save(W / "index" / "picker-embeddings.npy", (p / np.linalg.norm(p, axis=1, keepdims=True)).astype(np.float16))
    (W / "index" / "picker-rows.json").write_text(json.dumps([{"print": pr, "group": f"p{i % 2}"}
                                                             for i, pr in enumerate(prints)]))
    (W / "index" / "release.json").write_text(json.dumps({"scanner": "test", "cards": cards, "pictures": len(handles)}))
    return W
