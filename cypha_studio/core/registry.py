"""
cypha_studio.core.registry
──────────────────────────
Model registry: save, load, version, and promote trained models.

Every saved model is a directory:
  ~/.cypha/models/<name>/<version>/
    model.cypha          # binary state (cypha_save_binary format)
    preprocessor.json    # Preprocessor state
    card.json            # ModelCard metadata

"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import Preprocessor

_CYPHA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _CYPHA_ROOT not in sys.path:
    sys.path.insert(0, _CYPHA_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# ModelCard
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelCard:
    """Structured metadata for a saved model."""

    name         : str
    version      : str
    description  : str  = ''
    author       : str  = ''
    date         : str  = field(default_factory=lambda: time.strftime('%Y-%m-%d'))
    task         : str  = 'classification'
    model_type   : str  = 'CyphaDIF'

    # Architecture
    encoder_type : str  = 'VectorEncoder'
    feat_dim     : int  = 128
    field_dim    : int  = 160
    n_classes    : int  = 0
    class_labels : list[str] = field(default_factory=list)
    input_dim    : int  = 0

    # Training context
    dataset_name : str  = ''
    n_train      : int  = 0
    n_val        : int  = 0
    train_steps  : int  = 0
    training_time_s : float = 0.0

    # Evaluation metrics
    val_accuracy : float = 0.0
    val_f1       : float = 0.0
    val_r2       : float = 0.0
    val_rmse     : float = 0.0
    calibration_error : float = 0.0
    ood_auroc    : float = 0.0

    # Flags
    gh_protected : bool = False
    stage        : str  = 'dev'   # dev | staging | production

    # Paths (relative to model directory)
    model_file        : str = 'model.cypha'
    preprocessor_file : str = 'preprocessor.json'
    card_file         : str = 'card.json'

    # Intended use and limitations
    intended_use      : str = ''
    known_limitations : str = ''
    training_data_desc: str = ''


# ─────────────────────────────────────────────────────────────────────────────
# ModelRegistry
# ─────────────────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Manages a directory of saved models with semantic versioning.

    registry = ModelRegistry()
    registry.register(clf, card, preprocessor)
    clf, pre, card = registry.load('iris-classifier', '1.0.0')
    registry.list_models()
    registry.promote('iris-classifier', '1.0.0', to='production')
    """

    def __init__(self, root: str = '~/.cypha/models'):
        self.root = Path(os.path.expanduser(root))
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Directory helpers ────────────────────────────────────────────────────

    def _model_dir(self, name: str, version: str) -> Path:
        return self.root / name / version

    def _model_path(self, name: str, version: str) -> Path:
        return self._model_dir(name, version) / 'model.cypha'

    def _pre_path(self, name: str, version: str) -> Path:
        return self._model_dir(name, version) / 'preprocessor.json'

    def _card_path(self, name: str, version: str) -> Path:
        return self._model_dir(name, version) / 'card.json'

    # ── Register ─────────────────────────────────────────────────────────────

    def register(self, model, card: ModelCard,
                 preprocessor: Preprocessor | None = None,
                 overwrite: bool = False) -> Path:
        """
        Save a model + preprocessor + card to the registry.

        model: CyphaDIF, DIFRegressor, RFFRegressor, TwoStageDIFRegressor, or MKERegressor
        Returns path to the model directory.
        """
        from Cypha import cypha_save_binary

        model_dir = self._model_dir(card.name, card.version)
        if model_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Model {card.name}/{card.version} already exists. "
                f"Use overwrite=True or bump the version."
            )
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model binary
        state = model.save_state()
        cypha_save_binary(state, str(self._model_path(card.name, card.version)))

        # Save preprocessor
        if preprocessor is not None:
            pre_state = preprocessor.save_state()
            with open(self._pre_path(card.name, card.version), 'w') as f:
                json.dump(pre_state, f, indent=2)

        # Save model card
        card_dict = asdict(card)
        with open(self._card_path(card.name, card.version), 'w') as f:
            json.dump(card_dict, f, indent=2)

        return model_dir

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(self, name: str, version: str = 'latest') -> tuple[Any, Preprocessor | None, ModelCard]:
        """
        Load (model, preprocessor, card) from registry.

        If version='latest', loads the most recent version.
        """
        from Cypha import (
            cypha_load_binary,
        )

        if version == 'latest':
            version = self._latest_version(name)

        card = self.load_card(name, version)
        state = cypha_load_binary(str(self._model_path(name, version)))

        # Reconstruct model from card
        model = self._reconstruct_model(card, state)

        # Load preprocessor if present
        pre_path = self._pre_path(name, version)
        preprocessor = None
        if pre_path.exists():
            with open(pre_path) as f:
                pre_state = json.load(f)
            preprocessor = Preprocessor()
            preprocessor.load_state(pre_state)

        return model, preprocessor, card

    def _reconstruct_model(self, card: ModelCard, state: dict):
        """Reconstruct the right model class from card metadata and state."""
        from Cypha import (
            CyphaDIF,
            DIFRegressor,
            MKERegressor,
            RFFEncoder,
            RFFRegressor,
            TwoStageDIFRegressor,
            VectorEncoder,
        )

        if card.model_type == 'CyphaDIF':
            # Infer field_dim from saved state: field_W_T is (field_dim, field_dim)
            field_dim = card.field_dim
            if 'field_W_T' in state:
                wt = state['field_W_T']
                if hasattr(wt, 'shape'):
                    field_dim = wt.shape[0]
            if card.encoder_type == 'RFFEncoder':
                d = state.get('D', 256) if state.get('D') else 256
                enc = RFFEncoder(card.input_dim, D=d, gamma=1.0, seed=0)
            else:
                enc = VectorEncoder(card.input_dim)
            model = CyphaDIF(encoder=enc, field_dim=field_dim,
                             rng=np.random.default_rng(0))
            model.load_state(state)

        elif card.model_type == 'RFFRegressor':
            model = RFFRegressor()
            model.load_state(state)

        elif card.model_type == 'DIFRegressor':
            enc = VectorEncoder(card.input_dim)
            ne = int(state.get('n_experts', max(8, int(card.n_classes or 8))))
            tlr = float(state.get('target_lr', 0.06))
            model = DIFRegressor(enc, field_dim=card.field_dim, n_experts=ne,
                                 target_lr=tlr, rng=np.random.default_rng(0))
            model.load_state(state)

        elif card.model_type == 'TwoStageDIF':
            model = TwoStageDIFRegressor()
            model.load_state(state)

        elif card.model_type == 'MKE':
            enc = RFFEncoder(card.input_dim, D=state.get('D', 256), gamma=1.0, seed=0)
            cs = state.get('clf_state') or {}
            fd = int(card.field_dim) if card.field_dim else 160
            wt = cs.get('field_W_T')
            if wt is not None and hasattr(wt, 'shape'):
                fd = int(wt.shape[0])
            k = int(card.n_classes) if card.n_classes else 8
            model = MKERegressor(enc, K=k, field_dim=fd, rng=np.random.default_rng(0))
            model.load_state(state)

        else:
            raise ValueError(f"Unknown model_type in card: {card.model_type!r}")

        return model

    # ── Card operations ──────────────────────────────────────────────────────

    def load_card(self, name: str, version: str) -> ModelCard:
        card_path = self._card_path(name, version)
        if not card_path.exists():
            raise FileNotFoundError(f"No model card at {card_path}")
        with open(card_path) as f:
            d = json.load(f)
        return ModelCard(**{k: v for k, v in d.items()
                            if k in ModelCard.__dataclass_fields__})

    def update_card(self, name: str, version: str, **kwargs):
        """Update fields on an existing model card."""
        card = self.load_card(name, version)
        for k, v in kwargs.items():
            if hasattr(card, k):
                setattr(card, k, v)
        with open(self._card_path(name, version), 'w') as f:
            json.dump(asdict(card), f, indent=2)

    def promote(self, name: str, version: str,
                to: str = 'production') -> ModelCard:
        """Promote a model to staging or production."""
        card = self.load_card(name, version)
        card.stage = to
        with open(self._card_path(name, version), 'w') as f:
            json.dump(asdict(card), f, indent=2)
        return card

    # ── Listing ──────────────────────────────────────────────────────────────

    def list_model_names(self) -> list[str]:
        """Registry top-level model names (subdirectories); no ``card.json`` read."""
        if not self.root.exists():
            return []
        return sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        )

    def registered_versions(self, name: str) -> list[str]:
        """Version dirs under ``name`` that contain ``card.json``."""
        base = self.root / name
        if not base.is_dir():
            return []
        out: list[str] = []
        for vd in sorted(base.iterdir()):
            if vd.is_dir() and (vd / 'card.json').exists():
                out.append(vd.name)
        return out

    def registered_entry_count(self) -> int:
        """Number of (name, version) pairs with ``card.json`` (no card load)."""
        return sum(
            len(self.registered_versions(n)) for n in self.list_model_names()
        )

    def iter_registered_pairs(self) -> Iterator[tuple[str, str]]:
        """Yield ``(name, version)`` for each ``card.json`` under the registry root."""
        for name in self.list_model_names():
            for version in self.registered_versions(name):
                yield name, version

    def list_models(self) -> list[ModelCard]:
        """List all models across all versions."""
        cards = []
        for model_dir in sorted(self.root.iterdir()):
            if not model_dir.is_dir():
                continue
            for version_dir in sorted(model_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                card_path = version_dir / 'card.json'
                if card_path.exists():
                    try:
                        with open(card_path) as f:
                            d = json.load(f)
                        cards.append(ModelCard(**{k: v for k, v in d.items()
                                                  if k in ModelCard.__dataclass_fields__}))
                    except Exception:
                        pass
        return cards

    def list_versions(self, name: str) -> list[str]:
        model_root = self.root / name
        if not model_root.exists():
            return []
        return sorted(v.name for v in model_root.iterdir() if v.is_dir())

    def _latest_version(self, name: str) -> str:
        versions = self.list_versions(name)
        if not versions:
            raise FileNotFoundError(f"No versions found for model {name!r}")
        return versions[-1]

    def exists(self, name: str, version: str = 'latest') -> bool:
        if version == 'latest':
            return bool(self.list_versions(name))
        return self._model_dir(name, version).exists()

    # ── Delete ───────────────────────────────────────────────────────────────

    def delete(self, name: str, version: str):
        model_dir = self._model_dir(name, version)
        if model_dir.exists():
            shutil.rmtree(model_dir)
        # Remove parent if empty
        parent = self.root / name
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    # ── Compare ──────────────────────────────────────────────────────────────

    def compare(self, models: list[tuple[str, str]],
                test_ds=None) -> list[dict]:
        """
        Head-to-head comparison of multiple (name, version) pairs.
        If test_ds provided, evaluates each on it.
        Returns list of dicts with card fields + optional test metrics.
        """
        rows = []
        for name, version in models:
            try:
                card = self.load_card(name, version)
                row = {
                    'name'        : card.name,
                    'version'     : card.version,
                    'task'        : card.task,
                    'model_type'  : card.model_type,
                    'val_accuracy': card.val_accuracy,
                    'val_f1'      : card.val_f1,
                    'val_r2'      : card.val_r2,
                    'stage'       : card.stage,
                    'n_train'     : card.n_train,
                    'train_steps' : card.train_steps,
                }
                if test_ds is not None:
                    model, pre, _ = self.load(name, version)
                    from .trainer import Trainer
                    trainer = Trainer()
                    trainer._model = model
                    trainer._preprocessor = pre
                    metrics = trainer.evaluate(test_ds)
                    row['test_accuracy'] = metrics.accuracy
                    row['test_f1']       = metrics.macro_f1
                    row['test_r2']       = metrics.r2_score
                rows.append(row)
            except Exception as e:
                rows.append({'name': name, 'version': version, 'error': str(e)})
        return rows

    # ── Next version helper ───────────────────────────────────────────────────

    def next_version(self, name: str, bump: str = 'minor') -> str:
        """
        Suggest the next semantic version for a model name.
        bump: 'major', 'minor', 'patch'
        """
        versions = self.list_versions(name)
        if not versions:
            return '1.0.0'
        latest = versions[-1]
        try:
            parts = [int(x) for x in latest.split('.')]
            while len(parts) < 3:
                parts.append(0)
        except ValueError:
            return '1.0.0'

        if bump == 'major':
            parts[0] += 1; parts[1] = 0; parts[2] = 0
        elif bump == 'minor':
            parts[1] += 1; parts[2] = 0
        else:
            parts[2] += 1
        return '.'.join(str(p) for p in parts)
