"""
Lightweight model versioning registry.

Prevents overwriting good models during training runs by storing
each training result in a versioned subdirectory.
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Track and version trained model artifacts."""

    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / "registry.json"
        self._load_registry()

    # ── persistence ──────────────────────────────

    def _load_registry(self):
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": [], "active": None}

    def _save(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    # ── public API ───────────────────────────────

    def register_model(self, metrics: dict, description: str = "") -> str:
        """
        Snapshot current model artifacts into a versioned subdirectory.

        Args:
            metrics: dict like ``{'accuracy': 0.95, 'loss': 0.12}``
            description: optional human-readable note

        Returns:
            Version ID string (timestamp-based).
        """
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_dir = self.models_dir / f"v_{version}"
        versioned_dir.mkdir(exist_ok=True)

        for artifact in ("sign_model.keras", "label_encoder.pkl", "max_length.txt"):
            src = self.models_dir / artifact
            if src.exists():
                shutil.copy2(src, versioned_dir / artifact)

        entry = {
            "version": version,
            "metrics": metrics,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "path": str(versioned_dir),
        }
        self.registry["models"].append(entry)
        self._save()
        logger.info("Registered model v_%s  (accuracy=%.2f%%)", version, metrics.get("accuracy", 0) * 100)
        return version

    def promote(self, version: str):
        """Mark *version* as the active production model."""
        self.registry["active"] = version
        self._save()
        logger.info("Promoted model v_%s to active", version)

    def latest(self) -> dict | None:
        """Return the most recently registered entry, or None."""
        return self.registry["models"][-1] if self.registry["models"] else None
