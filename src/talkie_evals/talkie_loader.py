"""Pinned Talkie model loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from talkie_evals.constants import MODEL_REVISIONS


def load_talkie(model_name: str, cache_dir: str):
    """Load Talkie while pinning HF model snapshots.

    Upstream Talkie currently calls hf_hub_download without a revision argument,
    so we patch the function it imports before constructing the model.
    """
    import talkie.download as talkie_download
    import talkie.generate as talkie_generate
    from huggingface_hub import hf_hub_download
    from talkie import Talkie
    from talkie.config import MODELS

    def get_model_files_pinned(
        requested_model_name: str, cache_dir: str | Path | None = None
    ) -> tuple[Path, Path]:
        if requested_model_name not in MODELS:
            available = ", ".join(sorted(MODELS))
            raise ValueError(
                f"Unknown model {requested_model_name!r}. Available: {available}"
            )
        spec = MODELS[requested_model_name]
        kwargs: dict[str, Any] = {
            "repo_id": spec.repo_id,
            "revision": MODEL_REVISIONS[requested_model_name],
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        ckpt_path = Path(hf_hub_download(filename=spec.checkpoint_filename, **kwargs))
        vocab_path = Path(hf_hub_download(filename=spec.vocab_filename, **kwargs))
        return ckpt_path, vocab_path

    talkie_download.get_model_files = get_model_files_pinned
    talkie_generate.get_model_files = get_model_files_pinned
    return Talkie(model_name, cache_dir=cache_dir)
