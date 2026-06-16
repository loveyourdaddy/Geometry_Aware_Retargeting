"""
Interaction Mesh for spatial relationship preserving motion retargeting.
Ho et al., "Spatial Relationship Preserving Character Motion Adaptation", ACM TOG 2010.

핵심 수식 (레퍼런스: Physics-based-retargeting/interaction_mesh.py 방식):
    L(p_i) = p_i - Σ_j w_ij p_j   (uniform weight: w_ij = 1/deg)

char0(고정)와 char1(자유) 포인트를 합쳐 Delaunay 사면체화.
Laplacian 행렬을 자유/고정 블록으로 분리:
    L_ff: char1 ↔ char1 블록
    L_fa: char1 → char0(고정) 블록

char0가 고정일 때 char1 최적 위치는 LS 문제:
    min ||LB @ pos1 - (delta_src - LA @ pos0)||²
    → pos1_opt = LBpinv @ rhs   (1회 행렬곱)

런타임 비용: 프레임당 행렬곱 1회
"""

import numpy as np
from scipy.spatial import Delaunay
from collections import defaultdict


class InteractionMesh:
    """
    Interaction mesh: char0(고정 앵커) + char1(자유 변수) 합쳐 Delaunay 사면체화.

    레퍼런스 방식:
        - uniform weight (1/degree)
        - M_lap = pinv(LB) 사전 계산
        - 런타임: pos1_opt = M_lap @ (delta_src - LA @ pos0)
    """

    def __init__(self, positions_0, positions_1):
        """
        Args:
            positions_0: (T, J0=22, 3) — char0 글로벌 관절 위치 (고정 앵커)
            positions_1: (T, J1=22, 3) — char1 글로벌 관절 위치 (자유 변수)
        """
        T = positions_0.shape[0]
        self.n0 = positions_0.shape[1]  # char0 관절 수 (22)
        self.n1 = positions_1.shape[1]  # char1 관절 수 (22)
        self.n  = self.n0 + self.n1     # 전체 (44)

        self.src_positions = np.concatenate(
            [positions_0, positions_1], axis=1
        ).astype(np.float64)  # (T, 44, 3)

        # frame 0 topology 고정 (포즈에 크게 무관)
        self._build_laplacian_matrix(self.src_positions[0])

        # 소스 Laplacian 좌표 사전 계산: delta[t] = L @ src_positions[t]
        self.src_laplacian = np.einsum(
            'ij,tjk->tik', self.L, self.src_positions
        )  # (T, 44, 3)

        n_tet = Delaunay(self.src_positions[0]).simplices.shape[0]
        print(f"InteractionMesh: {self.n} vertices ({self.n0}+{self.n1}), "
              f"{T} frames, {n_tet} tetrahedra  |  "
              f"M_lap {self.M_lap.shape}")

    # ------------------------------------------------------------------
    # Laplacian 행렬 구축 (uniform weight, 레퍼런스 방식)
    # ------------------------------------------------------------------

    def _build_laplacian_matrix(self, positions):
        """
        Delaunay → 인접 리스트 → uniform Laplacian → M_lap 사전 계산.

        uniform weight: w_ij = 1 / deg(i)  (레퍼런스와 동일)

        행렬 분할:
            LA = L[:, :n0]   char0 열 (고정 앵커)
            LB = L[:, n0:]   char1 열 (자유 변수)

        사전 계산:
            M_lap = pinv(LB)   (n1 × n)
            런타임: pos1_opt = M_lap @ rhs,   rhs = delta_src - LA @ pos0
        """
        tri = Delaunay(positions)

        # 인접 리스트 (레퍼런스: defaultdict(set) 방식)
        adj = defaultdict(set)
        for tet in tri.simplices:
            for i in tet:
                for j in tet:
                    if i != j:
                        adj[i].add(j)

        # (n, n) Laplacian: L[i,i]=1, L[i,j]=-1/deg(i) for j in neighbors
        self.L = np.zeros((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            nbs = list(adj[i])
            self.L[i, i] = 1.0
            if not nbs:
                continue
            w = 1.0 / len(nbs)  # uniform weight (레퍼런스 방식)
            for j in nbs:
                self.L[i, j] = -w

        # 블록 분할
        self.LA  = self.L[:, :self.n0]  # (n, n0) char0 쪽
        self.LB  = self.L[:, self.n0:]  # (n, n1) char1 쪽
        self.LBT = self.LB.T            # 그래디언트용 (n1, n)

        # LS 사전 계산: M_lap = pinv(LB)  [n1, n]
        # 런타임: pos1_opt = M_lap @ (delta_src - LA @ pos0)
        self.M_lap = np.linalg.pinv(self.LB)  # (n1, n)

    # ------------------------------------------------------------------
    # 런타임: 닫힌 형식 Laplacian 최적 위치 계산
    # ------------------------------------------------------------------

    def compute_target_positions(self, pos0_t, delta_t):
        """
        Laplacian 보존 조건을 만족하는 char1 최적 위치 (1회 행렬곱).

        min_{pos1} || LB @ pos1 - rhs ||²
        → pos1_opt = M_lap @ rhs

        Args:
            pos0_t:  (n0, 3) char0 글로벌 위치 (고정)
            delta_t: (n, 3)  소스 Laplacian 좌표

        Returns:
            pos1_opt: (n1, 3) char1 Laplacian 최적 위치
        """
        rhs = delta_t - self.LA @ pos0_t  # (n, 3)
        return self.M_lap @ rhs            # (n1, 3)

    # ------------------------------------------------------------------
    # (하위 호환) L-BFGS용 에너지/그래디언트 — 필요시 유지
    # ------------------------------------------------------------------

    def compute_energy_and_grad(self, pos1_flat, pos0_t, delta_t):
        """
        E_L = || LA @ pos0 + LB @ pos1 - delta ||²
        grad = 2 * LB^T @ residual
        """
        pos1     = pos1_flat.reshape(self.n1, 3)
        residual = self.LA @ pos0_t + self.LB @ pos1 - delta_t
        energy   = float(np.sum(residual ** 2))
        grad     = 2.0 * (self.LBT @ residual)
        return energy, grad.flatten()
