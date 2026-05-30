"""Growing Neural Gas for auxiliary latent prototypes (Upgrade U1)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class GNGExpertManager:
    def __init__(
        self,
        d: int,
        eps_b: float = 0.05,
        eps_n: float = 0.006,
        lam: int = 100,
        age_max: int = 50,
        alpha_gng: float = 0.5,
        max_nodes: int = 256,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.d = int(d)
        self.eps_b = float(eps_b)
        self.eps_n = float(eps_n)
        self.lam = int(lam)
        self.age_max = int(age_max)
        self.alpha_gng = float(alpha_gng)
        self.max_nodes = int(max_nodes)
        self._rng = rng or np.random.default_rng(42)
        self._step_count = 0
        self._next_id = 0
        self.nodes: Dict[int, np.ndarray] = {}
        self.errors: Dict[int, float] = {}
        self.edges: Dict[Tuple[int, int], int] = {}
        self._init_two_nodes()

    def _init_two_nodes(self) -> None:
        for _ in range(2):
            nid = self._next_id
            self._next_id += 1
            self.nodes[nid] = self._rng.standard_normal(self.d) * 0.1
            self.errors[nid] = 0.0
        a, b = list(self.nodes.keys())[:2]
        self._add_edge(a, b)

    def _add_edge(self, i: int, j: int) -> None:
        if i == j:
            return
        key = (min(i, j), max(i, j))
        self.edges[key] = 0

    def _neighbors(self, i: int) -> List[int]:
        out: List[int] = []
        for (a, b), _ in self.edges.items():
            if a == i:
                out.append(b)
            elif b == i:
                out.append(a)
        return out

    def _two_closest(self, x: np.ndarray) -> Tuple[int, int]:
        x = np.asarray(x, dtype=np.float64).ravel()
        best_i, best_d = -1, float("inf")
        second_i, second_d = -1, float("inf")
        for nid, w in self.nodes.items():
            d = float(np.sum((x - w) ** 2))
            if d < best_d:
                second_i, second_d = best_i, best_d
                best_i, best_d = nid, d
            elif d < second_d:
                second_i, second_d = nid, d
        if second_i < 0:
            second_i = best_i
        return best_i, second_i

    def step(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.d:
            raise ValueError(f"Expected dim {self.d}, got {x.size}")
        bmu, bmu2 = self._two_closest(x)
        self.nodes[bmu] += self.eps_b * (x - self.nodes[bmu])
        for nb in self._neighbors(bmu):
            self.nodes[nb] += self.eps_n * (x - self.nodes[nb])
        err = float(np.sum((x - self.nodes[bmu]) ** 2))
        self.errors[bmu] = self.errors.get(bmu, 0.0) + err
        self._add_edge(bmu, bmu2)
        key = (min(bmu, bmu2), max(bmu, bmu2))
        self.edges[key] = 0
        for ekey in list(self.edges.keys()):
            if bmu in ekey:
                self.edges[ekey] += 1
        self._prune_old_edges()
        self._step_count += 1
        if self._step_count % self.lam == 0 and len(self.nodes) < self.max_nodes:
            self._insert_node()
        self._decay_errors()
        return bmu

    def _prune_old_edges(self) -> None:
        dead = [k for k, age in self.edges.items() if age > self.age_max]
        for k in dead:
            del self.edges[k]
        isolated = [
            nid
            for nid in list(self.nodes.keys())
            if len(self._neighbors(nid)) == 0 and len(self.nodes) > 2
        ]
        for nid in isolated:
            self._remove_node(nid)

    def _remove_node(self, nid: int) -> None:
        self.nodes.pop(nid, None)
        self.errors.pop(nid, None)
        for k in [k for k in self.edges if nid in k]:
            del self.edges[k]

    def _insert_node(self) -> None:
        if not self.errors:
            return
        q = max(self.errors, key=lambda i: self.errors[i])
        nbrs = self._neighbors(q)
        if not nbrs:
            return
        f = max(nbrs, key=lambda j: self.errors.get(j, 0.0))
        r = self._next_id
        self._next_id += 1
        self.nodes[r] = 0.5 * (self.nodes[q] + self.nodes[f])
        self.errors[r] = self.errors[q]
        self.errors[q] *= self.alpha_gng
        self.errors[f] *= self.alpha_gng
        self._add_edge(q, r)
        self._add_edge(r, f)
        old = (min(q, f), max(q, f))
        if old in self.edges:
            del self.edges[old]

    def _decay_errors(self) -> None:
        for nid in self.errors:
            self.errors[nid] *= 0.995

    def force_insert(self, node_id: int) -> None:
        nbrs = self._neighbors(node_id)
        if not nbrs or node_id not in self.nodes:
            self._insert_node()
            return
        f = nbrs[0]
        r = self._next_id
        self._next_id += 1
        self.nodes[r] = 0.5 * (self.nodes[node_id] + self.nodes[f])
        self.errors[r] = self.errors.get(node_id, 1.0)
        self._add_edge(node_id, r)

    def merge_with_nearest(self, node_id: int) -> None:
        if node_id not in self.nodes:
            return
        nbrs = self._neighbors(node_id)
        if not nbrs:
            return
        other = nbrs[0]
        self.nodes[other] = 0.5 * (self.nodes[other] + self.nodes[node_id])
        self._remove_node(node_id)

    def node_count(self) -> int:
        return len(self.nodes)

    def get_prototypes(self) -> np.ndarray:
        if not self.nodes:
            return np.zeros((0, self.d), dtype=np.float64)
        return np.stack(list(self.nodes.values()), axis=0)

    def context_bias(self, h: np.ndarray, labels: List[str], strength: float = 0.15) -> Dict[str, float]:
        """Soft bias toward labels whose delta is nearest a GNG node (auxiliary routing)."""
        if not labels or not self.nodes:
            return {lbl: 0.0 for lbl in labels}
        h = np.asarray(h, dtype=np.float64).ravel()
        protos = self.get_prototypes()
        dists = np.linalg.norm(protos - h, axis=1)
        bmu = int(np.argmin(dists))
        # spread bias inversely with distance to all nodes
        weights = np.exp(-dists / (float(np.median(dists)) + 1e-9))
        weights /= weights.sum() + 1e-12
        out: Dict[str, float] = {}
        for i, lbl in enumerate(labels):
            idx = i % len(weights)
            out[lbl] = float(strength * weights[idx])
        return out
