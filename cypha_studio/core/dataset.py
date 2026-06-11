"""
cypha_studio.core.dataset
─────────────────────────
Dataset loading, preprocessing, and streaming for Cypha training.

Supports:
  - CSV files (classification and regression)
  - Numpy .npy / .npz arrays
  - sklearn toy datasets
  - Streaming generators (online mode)

All datasets expose a stream() generator yielding (x_vector, label_str) tuples
compatible with CyphaDIF.train_step() / RFFRegressor.fit().
"""
from __future__ import annotations

import csv
import math
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor — fit on train, apply consistently at inference time
# ─────────────────────────────────────────────────────────────────────────────

class Preprocessor:
    """
    StandardScaler + optional PCA + optional RFF pre-encoding.

    Serialisable: save_state() / load_state() so the same transform
    applies at inference time after a model is loaded from the registry.
    """

    def __init__(self,
                 scale     : bool = True,
                 pca_dim   : int | None = None,
                 rff_dim   : int | None = None,
                 rff_gamma : float = 1.0,
                 seed      : int = 42):
        self.scale     = scale
        self.pca_dim   = pca_dim
        self.rff_dim   = rff_dim
        self.rff_gamma = rff_gamma
        self.seed      = seed

        self._mean : np.ndarray | None = None
        self._std  : np.ndarray | None = None
        self._pca_components : np.ndarray | None = None
        self._pca_mean       : np.ndarray | None = None
        self._rff_W : np.ndarray | None = None
        self._rff_b : np.ndarray | None = None
        self._fitted = False
        self._input_dim  : int | None = None
        self._output_dim : int | None = None

    def fit(self, X: np.ndarray) -> Preprocessor:
        X = np.asarray(X, dtype=np.float64)
        self._input_dim = X.shape[1]

        if self.scale:
            self._mean = X.mean(axis=0)
            self._std  = np.maximum(X.std(axis=0), 1e-8)
            X = (X - self._mean) / self._std

        if self.pca_dim is not None and self.pca_dim < X.shape[1]:
            self._pca_mean = X.mean(axis=0)
            Xc = X - self._pca_mean
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            self._pca_components = Vt[:self.pca_dim]
            X = Xc @ self._pca_components.T

        if self.rff_dim is not None:
            d = X.shape[1]
            rng = np.random.default_rng(self.seed)
            self._rff_W = rng.normal(0, self.rff_gamma, (self.rff_dim, d))
            self._rff_b = rng.uniform(0, 2 * math.pi, self.rff_dim)
            X = math.sqrt(2.0 / self.rff_dim) * np.cos(X @ self._rff_W.T + self._rff_b)

        self._output_dim = X.shape[1]
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=np.float64)
        if self.scale and self._mean is not None:
            X = (X - self._mean) / self._std
        if self._pca_components is not None:
            X = (X - self._pca_mean) @ self._pca_components.T
        if self._rff_W is not None:
            X = math.sqrt(2.0 / self.rff_dim) * np.cos(X @ self._rff_W.T + self._rff_b)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x.reshape(1, -1))[0]

    @property
    def output_dim(self) -> int | None:
        return self._output_dim

    def save_state(self) -> dict:
        return dict(
            scale=self.scale, pca_dim=self.pca_dim,
            rff_dim=self.rff_dim, rff_gamma=self.rff_gamma, seed=self.seed,
            mean=self._mean.tolist() if self._mean is not None else None,
            std=self._std.tolist()  if self._std  is not None else None,
            pca_components=self._pca_components.tolist() if self._pca_components is not None else None,
            pca_mean=self._pca_mean.tolist() if self._pca_mean is not None else None,
            rff_W=self._rff_W.tolist() if self._rff_W is not None else None,
            rff_b=self._rff_b.tolist() if self._rff_b is not None else None,
            fitted=self._fitted, input_dim=self._input_dim, output_dim=self._output_dim,
        )

    def load_state(self, state: dict) -> None:
        self.scale     = state['scale']
        self.pca_dim   = state['pca_dim']
        self.rff_dim   = state['rff_dim']
        self.rff_gamma = state['rff_gamma']
        self.seed      = state['seed']
        self._mean = np.array(state['mean']) if state['mean'] else None
        self._std  = np.array(state['std'])  if state['std']  else None
        self._pca_components = np.array(state['pca_components']) if state['pca_components'] else None
        self._pca_mean       = np.array(state['pca_mean'])       if state['pca_mean']       else None
        self._rff_W = np.array(state['rff_W']) if state['rff_W'] else None
        self._rff_b = np.array(state['rff_b']) if state['rff_b'] else None
        self._fitted     = state['fitted']
        self._input_dim  = state['input_dim']
        self._output_dim = state['output_dim']


# ─────────────────────────────────────────────────────────────────────────────
# DatasetStats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetStats:
    """Summary statistics computed from a dataset."""
    n_samples    : int = 0
    n_features   : int = 0
    n_classes    : int = 0
    class_counts : dict[str, int] = field(default_factory=dict)
    feature_means : list[float] | None = None
    feature_stds  : list[float] | None = None
    feature_mins  : list[float] | None = None
    feature_maxs  : list[float] | None = None
    class_balance : float = 0.0   # 1.0 = perfectly balanced, 0 = one class dominates
    missing_values : int  = 0


# ─────────────────────────────────────────────────────────────────────────────
# SplitConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SplitConfig:
    train_frac : float = 0.7
    val_frac   : float = 0.15
    test_frac  : float = 0.15
    shuffle    : bool  = True
    stratify   : bool  = True
    seed       : int   = 42

    def __post_init__(self):
        total = self.train_frac + self.val_frac + self.test_frac
        assert abs(total - 1.0) < 1e-6, f"Fractions must sum to 1.0, got {total}"


# ─────────────────────────────────────────────────────────────────────────────
# Base CyphaDataset
# ─────────────────────────────────────────────────────────────────────────────

class CyphaDataset:
    """
    Base dataset class. All datasets expose:
      - X  : (N, d) float64 array
      - y  : (N,)  string labels  OR  float targets
      - split(config) → (train_ds, val_ds, test_ds)
      - stream(split='train') → Generator[(x_vec, label_str)]
      - stats() → DatasetStats
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 name: str = "dataset",
                 feature_names: list[str] | None = None,
                 task: str = 'classification'):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y)
        self.name = name
        self.feature_names = feature_names or [f"f{i}" for i in range(self.X.shape[1])]
        self.task = task  # 'classification' or 'regression'

        # Split indices — populated by split()
        self._train_idx : np.ndarray | None = None
        self._val_idx   : np.ndarray | None = None
        self._test_idx  : np.ndarray | None = None

        # Preprocessor — fit on train split
        self.preprocessor : Preprocessor | None = None

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def labels(self) -> list[str]:
        return [str(v) for v in sorted(set(self.y))]

    @property
    def n_classes(self) -> int:
        return len(set(self.y))

    # ── Splitting ────────────────────────────────────────────────────────────

    def split(self, config: SplitConfig | None = None) -> tuple[CyphaDataset, CyphaDataset, CyphaDataset]:
        if config is None:
            config = SplitConfig()
        rng = np.random.default_rng(config.seed)
        N = self.n_samples
        idx = np.arange(N)

        if config.shuffle:
            rng.shuffle(idx)

        if config.stratify and self.task == 'classification':
            # Stratified split: maintain class proportions
            train_idx, val_idx, test_idx = [], [], []
            unique_labels = sorted(set(self.y))
            for lbl in unique_labels:
                lbl_idx = idx[self.y[idx] == lbl]
                n = len(lbl_idx)
                n_tr  = max(1, int(n * config.train_frac))
                n_val = max(1, int(n * config.val_frac))
                train_idx.extend(lbl_idx[:n_tr])
                val_idx.extend(lbl_idx[n_tr:n_tr + n_val])
                test_idx.extend(lbl_idx[n_tr + n_val:])
            train_idx = np.array(train_idx)
            val_idx   = np.array(val_idx)
            test_idx  = np.array(test_idx)
        else:
            n_tr  = int(N * config.train_frac)
            n_val = int(N * config.val_frac)
            train_idx = idx[:n_tr]
            val_idx   = idx[n_tr:n_tr + n_val]
            test_idx  = idx[n_tr + n_val:]

        self._train_idx = train_idx
        self._val_idx   = val_idx
        self._test_idx  = test_idx

        train_ds = CyphaDataset(self.X[train_idx], self.y[train_idx],
                                name=f"{self.name}_train",
                                feature_names=self.feature_names, task=self.task)
        val_ds   = CyphaDataset(self.X[val_idx],   self.y[val_idx],
                                name=f"{self.name}_val",
                                feature_names=self.feature_names, task=self.task)
        test_ds  = CyphaDataset(self.X[test_idx],  self.y[test_idx],
                                name=f"{self.name}_test",
                                feature_names=self.feature_names, task=self.task)
        return train_ds, val_ds, test_ds

    # ── Streaming ────────────────────────────────────────────────────────────

    def stream(self, shuffle: bool = False,
               seed: int = 42) -> Generator[tuple[np.ndarray, str], None, None]:
        """Yield (x_vector, label_string) pairs for train_step()."""
        idx = np.arange(self.n_samples)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        for i in idx:
            x = self.X[i]
            if self.preprocessor is not None:
                x = self.preprocessor.transform_one(x)
            yield x, str(self.y[i])

    def stream_xy(self, shuffle: bool = False,
                  seed: int = 42) -> Generator[tuple[np.ndarray, Any], None, None]:
        """Yield (x_vector, raw_y) for regression (y is float, not string)."""
        idx = np.arange(self.n_samples)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        for i in idx:
            x = self.X[i]
            if self.preprocessor is not None:
                x = self.preprocessor.transform_one(x)
            yield x, float(self.y[i])

    # ── Statistics ───────────────────────────────────────────────────────────

    def stats(self) -> DatasetStats:
        stats = DatasetStats()
        stats.n_samples  = self.n_samples
        stats.n_features = self.n_features
        stats.n_classes  = self.n_classes
        stats.feature_means = self.X.mean(axis=0).tolist()
        stats.feature_stds  = self.X.std(axis=0).tolist()
        stats.feature_mins  = self.X.min(axis=0).tolist()
        stats.feature_maxs  = self.X.max(axis=0).tolist()
        stats.missing_values = int(np.isnan(self.X).sum())

        if self.task == 'classification':
            for lbl in self.labels:
                stats.class_counts[lbl] = int((self.y == lbl).sum())
            counts = np.array(list(stats.class_counts.values()), dtype=float)
            n = counts.sum()
            if n > 0 and len(counts) > 1:
                p = counts / n
                uniform = 1.0 / len(counts)
                # Balance: 1 = uniform, 0 = single class
                stats.class_balance = 1.0 - float(np.abs(p - uniform).sum()) / 2.0

        return stats

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (f"CyphaDataset({self.name!r}, n={self.n_samples}, "
                f"d={self.n_features}, task={self.task!r})")


# ─────────────────────────────────────────────────────────────────────────────
# Concrete loaders
# ─────────────────────────────────────────────────────────────────────────────

class CSVDataset(CyphaDataset):
    """
    Load from CSV. Last column is the target by default.

    csv_loader = CSVDataset.from_file('data.csv', target_col='label')
    """

    @classmethod
    def from_file(cls, path: str | Path,
                  target_col : str | int = -1,
                  feature_cols: list[str | int] | None = None,
                  has_header : bool = True,
                  delimiter  : str = ',',
                  name       : str | None = None,
                  task       : str = 'classification',
                  read_chunk_rows: int | None = None) -> CSVDataset:
        """
        Load CSV into dense ``X``, ``y``.

        ``read_chunk_rows``: if > 0, stream the file in chunks of this many rows
        (lower peak memory than buffering every row in a Python list first).
        """
        path = Path(path)
        chunk = max(0, int(read_chunk_rows or 0))
        header: list[str] | None = None

        def _batch_to_arrays(
            batch: list[list[str]],
            target_idx: int,
            feat_indices: list[int],
        ) -> tuple[np.ndarray, np.ndarray]:
            X_raw = [[row[i] for i in feat_indices] for row in batch]
            y_raw = [row[target_idx] for row in batch]
            Xb = np.array([[float(v) for v in row] for row in X_raw],
                          dtype=np.float64)
            if task == 'regression':
                yb = np.array([float(v) for v in y_raw])
            else:
                yb = np.array(y_raw, dtype=str)
            return Xb, yb

        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            if has_header:
                header = next(reader)
            first: list[str] | None = None
            for row in reader:
                if row:
                    first = row
                    break
            if first is None:
                raise ValueError(f"No data in {path}")

            ncols = len(first)
            if isinstance(target_col, str):
                if header is None:
                    raise ValueError("target_col as string requires has_header=True")
                target_idx = header.index(target_col)
            else:
                target_idx = target_col if target_col >= 0 else ncols + target_col

            if feature_cols is not None:
                feat_indices = []
                for c in feature_cols:
                    if isinstance(c, str):
                        assert header is not None
                        feat_indices.append(header.index(c))
                    else:
                        feat_indices.append(c)
            else:
                feat_indices = [i for i in range(ncols) if i != target_idx]

            feat_names = ([header[i] for i in feat_indices] if header
                          else [f"f{i}" for i in feat_indices])

            if chunk <= 0:
                rest = [row for row in reader if row]
                rows = [first] + rest
                X, y = _batch_to_arrays(rows, target_idx, feat_indices)
            else:
                X_parts: list[np.ndarray] = []
                y_parts: list[np.ndarray] = []
                buf: list[list[str]] = [first]
                for row in reader:
                    if not row:
                        continue
                    buf.append(row)
                    if len(buf) >= chunk:
                        Xb, yb = _batch_to_arrays(buf, target_idx, feat_indices)
                        X_parts.append(Xb)
                        y_parts.append(yb)
                        buf.clear()
                if buf:
                    Xb, yb = _batch_to_arrays(buf, target_idx, feat_indices)
                    X_parts.append(Xb)
                    y_parts.append(yb)
                if len(X_parts) == 1:
                    X, y = X_parts[0], y_parts[0]
                else:
                    X = np.vstack(X_parts)
                    y = np.concatenate(y_parts)

        ds = cls(X, y, name=name or path.stem,
                 feature_names=feat_names, task=task)
        return ds


class NumpyDataset(CyphaDataset):
    """Load from .npy / .npz files."""

    @classmethod
    def from_arrays(cls, X: np.ndarray, y: np.ndarray,
                    name: str = 'numpy_dataset',
                    feature_names: list[str] | None = None,
                    task: str = 'classification') -> NumpyDataset:
        return cls(X, y, name=name, feature_names=feature_names, task=task)

    @classmethod
    def from_npz(cls, path: str | Path,
                 x_key: str = 'X', y_key: str = 'y',
                 task: str = 'classification') -> NumpyDataset:
        data = np.load(path, allow_pickle=True)
        return cls(data[x_key], data[y_key],
                   name=Path(path).stem, task=task)


class SklearnDataset(CyphaDataset):
    """Load from sklearn toy datasets."""

    @classmethod
    def load(cls, name: str,
             task: str = 'classification') -> SklearnDataset:
        """
        name: 'iris', 'wine', 'breast_cancer', 'digits', 'diabetes'
        """
        loaders = {
            'iris'          : ('sklearn.datasets', 'load_iris'),
            'wine'          : ('sklearn.datasets', 'load_wine'),
            'breast_cancer' : ('sklearn.datasets', 'load_breast_cancer'),
            'digits'        : ('sklearn.datasets', 'load_digits'),
            'diabetes'      : ('sklearn.datasets', 'load_diabetes'),
        }
        if name not in loaders:
            raise ValueError(f"Unknown dataset {name!r}. Available: {list(loaders)}")
        module_name, fn_name = loaders[name]
        import importlib
        module = importlib.import_module(module_name)
        bunch = getattr(module, fn_name)()

        X = bunch.data.astype(np.float64)
        if task == 'regression' and hasattr(bunch, 'target'):
            y = bunch.target.astype(float)
        elif hasattr(bunch, 'target_names'):
            y = np.array([bunch.target_names[t] for t in bunch.target])
        else:
            y = bunch.target.astype(str)

        feat_names = (list(bunch.feature_names)
                      if hasattr(bunch, 'feature_names') else None)
        return cls(X, y, name=name, feature_names=feat_names, task=task)


class StreamingDataset:
    """
    Wraps an arbitrary generator of (x, label) pairs.
    Does not materialise the full dataset in memory.
    """

    def __init__(self, generator_fn, name: str = 'stream'):
        self._generator_fn = generator_fn
        self.name = name

    def stream(self) -> Generator[tuple[np.ndarray, str], None, None]:
        yield from self._generator_fn()
