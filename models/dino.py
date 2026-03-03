import torch
import torch.nn as nn
from pathlib import Path

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        repo_dir = torch.hub._get_cache_or_reload(
            "facebookresearch/dinov2",
            force_reload=False,
            trust_repo=True,
            calling_fn="dino_wm",
            verbose=False,
            skip_validation=True,
        )
        self._patch_dinov2_for_py39(repo_dir)
        self.base_model = torch.hub.load(repo_dir, name, source="local")
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb

    @staticmethod
    def _patch_dinov2_for_py39(repo_dir):
        root = Path(repo_dir)
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            if "| None" not in text:
                continue
            if "from __future__ import annotations" in text:
                continue
            py.write_text("from __future__ import annotations\n" + text, encoding="utf-8")
