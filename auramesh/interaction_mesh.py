"""
Interaction Mesh for spatial relationship preserving motion retargeting.
Ho et al., "Spatial Relationship Preserving Character Motion Adaptation", ACM TOG 2010.

Builds a Delaunay tetrahedral mesh from joint positions of two interacting characters,
computes Laplacian coordinates, and provides energy/gradient for optimization.
"""

import numpy as np
from scipy.spatial import Delaunay


class InteractionMesh:
    """
    Interaction mesh built from concatenated joint positions of two characters.

    The topology is fixed from frame 0 (Delaunay tetrahedralization).
    Laplacian weights are inversely proportional to inter-vertex distance.
    Source Laplacian coordinates (delta) are precomputed for all frames.

    Optimization target:
        E_L = Σ_j || L(p'_j) - δ_j ||²
    where L(p_j) = p_j - Σ_l w_{jl} * p_l  (Laplacian coordinate)
    """

    def __init__(self, positions_0, positions_1):
        """
        Args:
            positions_0: (T, J0, 3) ndarray - char0 global joint positions per frame
            positions_1: (T, J1, 3) ndarray - char1 global joint positions per frame
        """
        T = positions_0.shape[0]
        self.n0 = positions_0.shape[1]
        self.n1 = positions_1.shape[1]
        self.n = self.n0 + self.n1

        # (T, n, 3)
        self.src_positions = np.concatenate([positions_0, positions_1], axis=1).astype(np.float64)

        # Build Laplacian matrix from frame 0 topology (kept fixed per paper)
        self._build_laplacian_matrix(self.src_positions[0])

        # Precompute source Laplacian coordinates: delta[t] = L @ src_positions[t]
        # shape: (T, n, 3)
        self.src_laplacian = np.einsum('ij,tjk->tik', self.L, self.src_positions)

        print(f"InteractionMesh: {self.n} vertices ({self.n0} + {self.n1}), "
              f"{T} frames, {int((self.L != 0).sum() - self.n)} off-diagonal edges")

    def _build_laplacian_matrix(self, positions):
        """
        Delaunay tetrahedralization of frame-0 positions → Laplacian matrix L.
        L[j,j] = 1,  L[j,l] = -w_{jl}  for neighbors l of j.
        Weights: normalized inverse-distance.
        """
        tri = Delaunay(positions)

        # Adjacency from simplices
        neighbors = [set() for _ in range(self.n)]
        for simplex in tri.simplices:
            for i in simplex:
                for j in simplex:
                    if i != j:
                        neighbors[i].add(j)

        # Build (n, n) Laplacian matrix
        self.L = np.zeros((self.n, self.n), dtype=np.float64)
        for j in range(self.n):
            nbs = sorted(neighbors[j])
            self.L[j, j] = 1.0
            if not nbs:
                continue
            dists = np.linalg.norm(positions[nbs] - positions[j], axis=-1)
            dists = np.maximum(dists, 1e-8)
            w = 1.0 / dists
            w /= w.sum()
            for nb, wi in zip(nbs, w):
                self.L[j, nb] = -wi

        # Partition into char0 (fixed) and char1 (free) columns
        # residual = LA @ pos0 + LB @ pos1 - delta
        self.LA = self.L[:, :self.n0]   # (n, n0)
        self.LB = self.L[:, self.n0:]   # (n, n1)
        # Precompute LB^T for gradient: grad = 2 * LB^T @ residual
        self.LBT = self.LB.T            # (n1, n)

    def compute_energy_and_grad(self, pos1_flat, pos0_t, delta_t):
        """
        Laplacian energy and gradient for one frame.

        E_L = || LA @ pos0 + LB @ pos1 - delta ||²
        ∂E_L/∂pos1 = 2 * LB^T @ residual

        Args:
            pos1_flat: (n1*3,) - char1 positions flattened (optimization variable)
            pos0_t:    (n0, 3) - char0 positions (fixed, from source motion)
            delta_t:   (n, 3)  - source Laplacian coordinates at this frame

        Returns:
            energy: float
            grad:   (n1*3,) ndarray
        """
        pos1 = pos1_flat.reshape(self.n1, 3)

        residual = self.LA @ pos0_t + self.LB @ pos1 - delta_t  # (n, 3)
        energy = float(np.sum(residual ** 2))
        grad = 2.0 * (self.LBT @ residual)                      # (n1, 3)

        return energy, grad.flatten()
