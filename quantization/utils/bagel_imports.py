"""
量化共用组件的统一导入接口。

提供 HybridQuantLinear、LinearCKA、CombinedSimilarity、ALGORITHM_POOL 等核心组件。
HybridQuantLinear 已复制到本地 layers/ 目录，其余组件
直接在本文件定义，彻底消除对 Bagel 运行时 import 链的依赖。
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------
# HybridQuantLinear: 从本地 layers/ 导入
# ------------------------------------------------------------------
_QUANT_ROOT = str(Path(__file__).resolve().parent.parent)
if _QUANT_ROOT not in sys.path:
    sys.path.insert(0, _QUANT_ROOT)

from layers.hybrid_quant_linear import HybridQuantLinear  # noqa: E402, F401


# ------------------------------------------------------------------
# LinearCKA
# ------------------------------------------------------------------

class LinearCKA:
    """
    线性中心化核对齐 (Linear Centered Kernel Alignment)

    CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    其中 K = X @ X^T, L = Y @ Y^T (线性核)
    """

    @staticmethod
    def compute(X: torch.Tensor, Y: torch.Tensor) -> float:
        if X.shape[0] != Y.shape[0]:
            min_n = min(X.shape[0], Y.shape[0])
            X = X[:min_n]
            Y = Y[:min_n]

        X = X.float()
        Y = Y.float()
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        K = X @ X.T
        L = Y @ Y.T

        hsic_kl = torch.sum(K * L).item()
        hsic_kk = torch.norm(K, p='fro').item() ** 2
        hsic_ll = torch.norm(L, p='fro').item() ** 2

        denom = (hsic_kk * hsic_ll) ** 0.5
        if denom < 1e-10:
            return 0.0
        return min(hsic_kl / denom, 1.0)

    @staticmethod
    def compute_batched(
        X_batches: List[torch.Tensor],
        Y_batches: List[torch.Tensor],
        subsample_step: int = 5,
    ) -> float:
        """Per-sample CKA then average."""
        cka_scores = []

        for X_batch, Y_batch in zip(X_batches, Y_batches):
            if X_batch.dim() == 3:
                X_batch = X_batch[:, ::subsample_step, :]
                X_batch = X_batch.reshape(-1, X_batch.shape[-1])
            if Y_batch.dim() == 3:
                Y_batch = Y_batch[:, ::subsample_step, :]
                Y_batch = Y_batch.reshape(-1, Y_batch.shape[-1])

            if X_batch.shape[0] != Y_batch.shape[0]:
                min_n = min(X_batch.shape[0], Y_batch.shape[0])
                X_batch = X_batch[:min_n]
                Y_batch = Y_batch[:min_n]

            if X_batch.shape[0] < 2:
                continue

            X_batch = X_batch.float()
            Y_batch = Y_batch.float()
            X_batch = X_batch - X_batch.mean(dim=0, keepdim=True)
            Y_batch = Y_batch - Y_batch.mean(dim=0, keepdim=True)

            K = X_batch @ X_batch.T
            L = Y_batch @ Y_batch.T

            hsic_kl = torch.sum(K * L).item()
            hsic_kk = torch.norm(K, p='fro').item() ** 2
            hsic_ll = torch.norm(L, p='fro').item() ** 2

            denom = (hsic_kk * hsic_ll) ** 0.5
            if denom > 1e-10:
                cka_scores.append(min(hsic_kl / denom, 1.0))

            del K, L, X_batch, Y_batch

        if not cka_scores:
            return 0.0
        return sum(cka_scores) / len(cka_scores)


# ------------------------------------------------------------------
# CombinedSimilarity — CKA + Cosine + MSE 三合一指标
# ------------------------------------------------------------------

class CombinedSimilarity:
    """
    组合相似度指标，同时衡量三个互补维度：

      1. Linear CKA    — 全局表示结构相似性（对 isotropic scaling 不变）
      2. Cosine Sim     — 逐 token 方向对齐（捕捉方向偏移）
      3. Normalized MSE — 幅度保真度（捕捉尺度变化）

    score = w_cka * CKA + w_cos * CosineSim + w_mse * (1 / (1 + relative_mse))

    三者范围均在 [0, 1]，值越高越好。
    """

    @staticmethod
    def _prepare_batch(
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        subsample_step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if X_batch.dim() == 3:
            X_batch = X_batch[:, ::subsample_step, :]
            X_batch = X_batch.reshape(-1, X_batch.shape[-1])
        if Y_batch.dim() == 3:
            Y_batch = Y_batch[:, ::subsample_step, :]
            Y_batch = Y_batch.reshape(-1, Y_batch.shape[-1])

        if X_batch.shape[0] != Y_batch.shape[0]:
            min_n = min(X_batch.shape[0], Y_batch.shape[0])
            X_batch = X_batch[:min_n]
            Y_batch = Y_batch[:min_n]

        return X_batch.float(), Y_batch.float()

    @staticmethod
    def _cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        Xc = X - X.mean(dim=0, keepdim=True)
        Yc = Y - Y.mean(dim=0, keepdim=True)
        K = Xc @ Xc.T
        L = Yc @ Yc.T
        hsic_kl = torch.sum(K * L).item()
        hsic_kk = torch.norm(K, p='fro').item() ** 2
        hsic_ll = torch.norm(L, p='fro').item() ** 2
        denom = (hsic_kk * hsic_ll) ** 0.5
        if denom < 1e-10:
            return 0.0
        return min(hsic_kl / denom, 1.0)

    @staticmethod
    def _cosine(X: torch.Tensor, Y: torch.Tensor) -> float:
        sim = F.cosine_similarity(X, Y, dim=-1)
        return sim.mean().clamp(0.0, 1.0).item()

    @staticmethod
    def _mse_score(X: torch.Tensor, Y: torch.Tensor) -> float:
        mse = ((X - Y) ** 2).mean().item()
        x_var = (X ** 2).mean().item()
        relative_mse = mse / (x_var + 1e-10)
        return 1.0 / (1.0 + relative_mse)

    @staticmethod
    def compute_batched(
        X_batches: List[torch.Tensor],
        Y_batches: List[torch.Tensor],
        subsample_step: int = 5,
        cka_weight: float = 0.4,
        cosine_weight: float = 0.3,
        mse_weight: float = 0.3,
    ) -> float:
        """
        对多个 batch 计算组合相似度后取平均。
        接口与 LinearCKA.compute_batched 兼容（额外权重参数有默认值）。
        """
        scores = []

        for X_raw, Y_raw in zip(X_batches, Y_batches):
            X, Y = CombinedSimilarity._prepare_batch(X_raw, Y_raw, subsample_step)
            if X.shape[0] < 2:
                continue

            cka_val = CombinedSimilarity._cka(X, Y)
            cos_val = CombinedSimilarity._cosine(X, Y)
            mse_val = CombinedSimilarity._mse_score(X, Y)

            combined = (cka_weight * cka_val
                        + cosine_weight * cos_val
                        + mse_weight * mse_val)
            scores.append(combined)

            del X, Y

        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @staticmethod
    def compute_batched_detailed(
        X_batches: List[torch.Tensor],
        Y_batches: List[torch.Tensor],
        subsample_step: int = 5,
        cka_weight: float = 0.4,
        cosine_weight: float = 0.3,
        mse_weight: float = 0.3,
    ) -> Dict[str, float]:
        """返回各分项及综合得分，用于日志分析。"""
        all_cka, all_cos, all_mse, all_combined = [], [], [], []

        for X_raw, Y_raw in zip(X_batches, Y_batches):
            X, Y = CombinedSimilarity._prepare_batch(X_raw, Y_raw, subsample_step)
            if X.shape[0] < 2:
                continue
            cka_val = CombinedSimilarity._cka(X, Y)
            cos_val = CombinedSimilarity._cosine(X, Y)
            mse_val = CombinedSimilarity._mse_score(X, Y)
            combined = (cka_weight * cka_val
                        + cosine_weight * cos_val
                        + mse_weight * mse_val)
            all_cka.append(cka_val)
            all_cos.append(cos_val)
            all_mse.append(mse_val)
            all_combined.append(combined)
            del X, Y

        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        return {
            'combined': _avg(all_combined),
            'cka': _avg(all_cka),
            'cosine': _avg(all_cos),
            'mse_score': _avg(all_mse),
        }


# ------------------------------------------------------------------
# ALGORITHM_POOL
# ------------------------------------------------------------------
# 设计原则:
#   1. A (act_bit) 统一为 4，W (weight_bit) 可以是 4/3/2
#   2. 每个 W 档位有完全相同的 10 种算法组合
#   3. RTN 系列 (6种): rtn / smooth+rtn / awq+rtn / svd+rtn / smooth+svd+rtn / awq+svd+rtn
#   4. GPTQ 系列 (4种): gptq / smooth+gptq / svd+gptq / smooth+svd+gptq (SVDQuant)

_BASE = {
    'weight_bit': 4, 'act_bit': 4, 'act_unsigned': True,
    'use_smoothquant': False, 'smoothquant_alpha': 0.5,
    'use_awq': False, 'awq_alpha': 0.5, 'awq_n_grid': 20,
    'use_svd': False, 'svd_rank': 32,
    'use_sparse': False, 'sparse_threshold': 0.0001,
    'use_gptq': False,
    'gptq_group_size': 64, 'gptq_damp_percentage': 0.01, 'gptq_block_size': 128,
    'use_block_quant': False, 'use_block_quant_act': False,
    'block_size_weight': 256, 'block_size_act': 256,
}


def _algo(name, desc, **overrides):
    cfg = dict(_BASE)
    cfg.update(overrides)
    return {'name': name, 'description': desc, 'config': cfg}


def _build_pool():
    pool = {}
    for wb in [4, 3, 2]:
        s = f'w{wb}a4'
        W = f'W{wb}A4'
        w = {'weight_bit': wb}
        # RTN 系列
        pool[f'rtn_{s}'] = _algo(f'RTN {W}', f'Round-To-Nearest {W}', **w)
        pool[f'smooth_rtn_{s}'] = _algo(f'Smooth+RTN {W}', f'SmoothQuant+RTN {W}', **w, use_smoothquant=True)
        pool[f'awq_rtn_{s}'] = _algo(f'AWQ+RTN {W}', f'AWQ+RTN {W}', **w, use_awq=True)
        pool[f'svd_rtn_{s}'] = _algo(f'SVD+RTN {W}', f'SVD rank-32+RTN {W}', **w, use_svd=True)
        pool[f'smooth_svd_rtn_{s}'] = _algo(f'Smooth+SVD+RTN {W}', f'Smooth+SVD+RTN {W}', **w, use_smoothquant=True, use_svd=True)
        pool[f'awq_svd_rtn_{s}'] = _algo(f'AWQ+SVD+RTN {W}', f'AWQ+SVD+RTN {W}', **w, use_awq=True, use_svd=True)
        # GPTQ 系列
        pool[f'gptq_{s}'] = _algo(f'GPTQ {W}', f'GPTQ {W}', **w, use_gptq=True)
        pool[f'smooth_gptq_{s}'] = _algo(f'Smooth+GPTQ {W}', f'SmoothQuant+GPTQ {W}', **w, use_smoothquant=True, use_gptq=True)
        pool[f'svd_gptq_{s}'] = _algo(f'SVD+GPTQ {W}', f'SVD rank-32, GPTQ on residual {W}', **w, use_svd=True, use_gptq=True)
        pool[f'smooth_svd_gptq_{s}'] = _algo(f'SVDQuant {W}', f'Smooth+SVD+GPTQ (SVDQuant) {W}', **w, use_smoothquant=True, use_svd=True, use_gptq=True)
    return pool


ALGORITHM_POOL = _build_pool()
