#!/usr/bin/env python3


from __future__ import absolute_import, division, print_function

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

try:
    from scipy import sparse as scipy_sparse
except Exception as exc:  # pragma: no cover - reported clearly at runtime
    scipy_sparse = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


UNKNOWN_NORMALIZED_DISTANCE = np.float32(1.0 + 1.0e-6)




@dataclass
class SparseGLSAGlobalContext:
    """Reusable result of the unchanged global re-ranking stage."""

    raw_dist: np.ndarray
    reranked_qg: np.ndarray
    query_num: int
    gallery_num: int
    k1: int
    k2: int
    profile: dict

    @property
    def total_num(self):
        return self.query_num + self.gallery_num


def _as_device(device):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 运行稀疏 GLSA，但当前 CUDA 不可用")
    return device


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_global_features(prob_fea, gal_fea):
    if not isinstance(prob_fea, torch.Tensor) or not isinstance(
        gal_fea, torch.Tensor
    ):
        raise TypeError("probFea 和 galFea 必须是 torch.Tensor")
    if prob_fea.ndim != 2 or gal_fea.ndim != 2:
        raise ValueError("probFea 和 galFea 必须为 [样本数, 特征维度]")
    if prob_fea.shape[1] != gal_fea.shape[1]:
        raise ValueError("query/gallery 全局特征维度必须一致")
    if prob_fea.shape[0] <= 0 or gal_fea.shape[0] <= 0:
        raise ValueError("query 和 gallery 均不能为空")
    if not torch.isfinite(prob_fea).all() or not torch.isfinite(gal_fea).all():
        raise ValueError("全局特征包含 NaN 或 Inf")


def _validate_local_features(qf_local, gf_local, query_num, gallery_num):
    if not isinstance(qf_local, torch.Tensor) or not isinstance(
        gf_local, torch.Tensor
    ):
        raise TypeError("qf_local 和 gf_local 必须是 torch.Tensor")
    if qf_local.ndim != 3 or gf_local.ndim != 3:
        raise ValueError("qf_local 和 gf_local 必须为 [样本数, D, N]")
    if qf_local.shape[0] != query_num or gf_local.shape[0] != gallery_num:
        raise ValueError("全局与局部特征的 query/gallery 数量不一致")
    if qf_local.shape[1:] != gf_local.shape[1:]:
        raise ValueError("query/gallery 局部特征的 D、N 必须一致")
    if qf_local.shape[1] <= 0 or qf_local.shape[2] <= 0:
        raise ValueError("局部特征维度和 patch 数必须为正数")
    if not torch.isfinite(qf_local).all() or not torch.isfinite(gf_local).all():
        raise ValueError("局部特征包含 NaN 或 Inf")


def _legacy_stage_rerank_fast_exact(
    original_dist, query_num, all_num, k1, k2, lambda_value
):
    """Unmodified Legacy-Full k-reciprocal stage used by the fast wrapper."""
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(
        original_dist / np.max(original_dist, axis=0)
    )
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[
            forward_k_neigh_index, :k1 + 1
        ]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[
                candidate, :int(np.around(k1 / 2)) + 1
            ]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index,
                :int(np.around(k1 / 2)) + 1,
            ]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate
            )[0]
            candidate_k_reciprocal_index = (
                candidate_forward_k_neigh_index[fi_candidate]
            )
            if len(
                np.intersect1d(
                    candidate_k_reciprocal_index, k_reciprocal_index
                )
            ) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index,
                    candidate_k_reciprocal_index,
                )

        k_reciprocal_expansion_index = np.unique(
            k_reciprocal_expansion_index
        )
        weight = np.exp(
            -original_dist[i, k_reciprocal_expansion_index]
        )
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(
                V[initial_rank[i, :k2], :], axis=0
            )
        V = V_qe
        del V_qe
    del initial_rank

    inv_index = []
    for i in range(gallery_num):
        inv_index.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
    for i in range(query_num):
        temp_min = np.zeros(
            shape=[1, gallery_num], dtype=np.float16
        )
        nonzero = np.where(V[i, :] != 0)[0]
        images = [inv_index[index] for index in nonzero]
        for j in range(len(nonzero)):
            temp_min[0, images[j]] = temp_min[0, images[j]] + np.minimum(
                V[i, nonzero[j]], V[images[j], nonzero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = (
        jaccard_dist * (1 - lambda_value)
        + original_dist * lambda_value
    )
    del original_dist, V, jaccard_dist
    return final_dist[:query_num, query_num:]



@torch.no_grad()
def prepare_glsa_global_context(probFea, galFea, k1=20, k2=6):
    """Compute the legacy global distance and k-reciprocal stage once.

    The returned context can be reused for several Top-M settings.  Its
    ``raw_dist`` is the complete T-by-T global squared-Euclidean matrix used
    both by the legacy global stage and by sparse candidate selection.
    """

    _validate_global_features(probFea, galFea)
    if isinstance(k1, bool) or int(k1) != k1 or int(k1) <= 0:
        raise ValueError("k1 必须是正整数")
    if isinstance(k2, bool) or int(k2) != k2 or int(k2) <= 0:
        raise ValueError("k2 必须是正整数")
    k1, k2 = int(k1), int(k2)

    query_num = int(probFea.shape[0])
    gallery_num = int(galFea.shape[0])
    total_num = query_num + gallery_num
    if k1 >= total_num:
        raise ValueError("k1 必须小于 query+gallery 样本总数")
    if k2 > total_num:
        raise ValueError("k2 不能大于 query+gallery 样本总数")

    total_start = time.perf_counter()
    distance_start = time.perf_counter()
    features = torch.cat([probFea, galFea], dim=0)
    squared_norm = torch.pow(features, 2).sum(dim=1, keepdim=True)
    distance = squared_norm.expand(total_num, total_num).clone()
    distance.add_(squared_norm.t())
    distance.addmm_(features, features.t(), beta=1.0, alpha=-2.0)
    raw_dist = distance.detach().cpu().numpy().astype(np.float32, copy=False)
    del features, squared_norm, distance
    global_distance_s = time.perf_counter() - distance_start

    rerank_start = time.perf_counter()
    reranked_qg = _legacy_stage_rerank_fast_exact(
        raw_dist,
        query_num,
        total_num,
        k1,
        k2,
        lambda_value=0.3,
    )
    global_reranking_s = time.perf_counter() - rerank_start
    profile = {
        "global_distance_s": float(global_distance_s),
        "global_reranking_s": float(global_reranking_s),
        "global_postprocess_s": float(time.perf_counter() - total_start),
    }
    return SparseGLSAGlobalContext(
        raw_dist=raw_dist,
        reranked_qg=reranked_qg,
        query_num=query_num,
        gallery_num=gallery_num,
        k1=k1,
        k2=k2,
        profile=profile,
    )


def _validate_context(context, query_num, gallery_num, k1, k2):
    if not isinstance(context, SparseGLSAGlobalContext):
        raise TypeError("global_context 必须由 prepare_glsa_global_context() 创建")
    if context.query_num != query_num or context.gallery_num != gallery_num:
        raise ValueError("global_context 与当前 query/gallery 数量不一致")
    if context.k1 != int(k1) or context.k2 != int(k2):
        raise ValueError("global_context 的 k1/k2 与当前调用不一致")
    total_num = query_num + gallery_num
    if context.raw_dist.shape != (total_num, total_num):
        raise ValueError("global_context.raw_dist 形状错误")
    if context.reranked_qg.shape != (query_num, gallery_num):
        raise ValueError("global_context.reranked_qg 形状错误")
    if not np.all(np.isfinite(context.raw_dist)):
        raise ValueError("global_context.raw_dist 包含 NaN 或 Inf")


def build_sparse_candidate_graph(
    raw_global_dist,
    top_m,
    calibration_count=16,
    device=None,
    row_batch_size=256,
    progress_callback=None,
):
    """Select and symmetrize all-node global Top-M candidate pairs.

    Returns unique unordered pairs ``left < right``.  The farthest calibration
    pairs are included in the same list so their true local distances can be
    used to estimate each row's legacy normalization scale.
    """

    if not isinstance(raw_global_dist, np.ndarray):
        raise TypeError("raw_global_dist 必须是 numpy.ndarray")
    if raw_global_dist.ndim != 2 or raw_global_dist.shape[0] != raw_global_dist.shape[1]:
        raise ValueError("raw_global_dist 必须是方阵")
    if not np.all(np.isfinite(raw_global_dist)):
        raise ValueError("raw_global_dist 包含 NaN 或 Inf")
    total_num = int(raw_global_dist.shape[0])
    if isinstance(top_m, bool) or not isinstance(top_m, (int, np.integer)):
        raise TypeError("top_m 必须是整数")
    top_m = int(top_m)
    if top_m <= 0 or top_m >= total_num:
        raise ValueError("top_m 必须位于 [1, query+gallery-1]")
    if isinstance(calibration_count, bool) or not isinstance(
        calibration_count, (int, np.integer)
    ):
        raise TypeError("calibration_count 必须是非负整数")
    calibration_count = int(calibration_count)
    if calibration_count < 0:
        raise ValueError("calibration_count 不能为负数")
    calibration_count = min(calibration_count, total_num - 1)
    if isinstance(row_batch_size, bool) or int(row_batch_size) != row_batch_size:
        raise TypeError("row_batch_size 必须是正整数")
    row_batch_size = int(row_batch_size)
    if row_batch_size <= 0:
        raise ValueError("row_batch_size 必须是正整数")
    device = _as_device(device)

    start_time = time.perf_counter()
    if top_m == total_num - 1:
        left, right = np.triu_indices(total_num, k=1)
        left = left.astype(np.int32, copy=False)
        right = right.astype(np.int32, copy=False)
        profile = {
            "candidate_construction_s": float(time.perf_counter() - start_time),
            "directed_near_candidates": int(total_num * top_m),
            "directed_calibration_candidates": 0,
            "computed_pair_count": int(left.size),
            "full_pair_count": int(total_num * (total_num - 1) // 2),
            "computed_pair_fraction": 1.0,
        }
        return left, right, profile

    key_parts = []
    directed_calibration = 0
    for row_start in range(0, total_num, row_batch_size):
        row_stop = min(row_start + row_batch_size, total_num)
        rows_np = np.arange(row_start, row_stop, dtype=np.int64)
        block_np = np.ascontiguousarray(raw_global_dist[row_start:row_stop])
        block = torch.from_numpy(block_np)
        if device.type == "cpu":
            block = block.clone()
        else:
            block = block.to(device=device, non_blocking=False)
        local_rows = torch.arange(row_stop - row_start, device=device)
        global_rows = torch.arange(row_start, row_stop, device=device)

        block[local_rows, global_rows] = float("inf")
        near = torch.topk(
            block, k=top_m, dim=1, largest=False, sorted=False
        ).indices.cpu().numpy().astype(np.int64, copy=False)
        near_rows = np.repeat(rows_np, top_m)
        near_cols = near.reshape(-1)
        near_low = np.minimum(near_rows, near_cols)
        near_high = np.maximum(near_rows, near_cols)
        key_parts.append(near_low * total_num + near_high)

        if calibration_count > 0:
            block[local_rows, global_rows] = float("-inf")
            far = torch.topk(
                block,
                k=calibration_count,
                dim=1,
                largest=True,
                sorted=False,
            ).indices.cpu().numpy().astype(np.int64, copy=False)
            far_rows = np.repeat(rows_np, calibration_count)
            far_cols = far.reshape(-1)
            far_low = np.minimum(far_rows, far_cols)
            far_high = np.maximum(far_rows, far_cols)
            key_parts.append(far_low * total_num + far_high)
            directed_calibration += int(far_rows.size)

        del block, near
        if progress_callback is not None:
            progress_callback("candidates", row_stop, total_num)

    _synchronize(device)
    keys = np.unique(np.concatenate(key_parts).astype(np.int64, copy=False))
    left = (keys // total_num).astype(np.int32, copy=False)
    right = (keys % total_num).astype(np.int32, copy=False)
    if np.any(left >= right):
        raise RuntimeError("候选图内部错误：无向边未满足 left < right")
    full_pair_count = total_num * (total_num - 1) // 2
    profile = {
        "candidate_construction_s": float(time.perf_counter() - start_time),
        "directed_near_candidates": int(total_num * top_m),
        "directed_calibration_candidates": int(directed_calibration),
        "computed_pair_count": int(left.size),
        "full_pair_count": int(full_pair_count),
        "computed_pair_fraction": float(left.size / float(full_pair_count)),
    }
    return left, right, profile


def _aggregate_patch_minima(minima, k_selected):
    """Legacy Top-K L2 aggregation, using its smaller complement when useful."""

    npatch = int(minima.shape[-1])
    if k_selected > npatch // 2:
        drop_count = npatch - k_selected
        all_squared = torch.sum(minima * minima, dim=-1)
        if drop_count:
            largest = torch.topk(
                minima, k=drop_count, dim=-1, largest=True, sorted=False
            ).values
            kept_squared = all_squared - torch.sum(largest * largest, dim=-1)
        else:
            kept_squared = all_squared
        return torch.sqrt(torch.clamp(kept_squared, min=0.0))
    selected = torch.topk(
        minima, k=k_selected, dim=-1, largest=False, sorted=False
    ).values
    return torch.sqrt(torch.sum(selected * selected, dim=-1))


@torch.no_grad()
def compute_sparse_edge_local_distances(
    qf_local,
    gf_local,
    left,
    right,
    k_per=0.9,
    pair_batch_size=256,
    device=None,
    progress_callback=None,
):
    """Compute exact bidirectional patch matching only for selected pairs."""

    query_num = int(qf_local.shape[0])
    gallery_num = int(gf_local.shape[0])
    _validate_local_features(qf_local, gf_local, query_num, gallery_num)
    total_num = query_num + gallery_num
    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("left/right 必须是相同长度的一维数组")
    if left.size and (
        np.min(left) < 0
        or np.max(right) >= total_num
        or np.any(left >= right)
    ):
        raise ValueError("left/right 包含非法或非规范化的无向边")
    if not 0.0 < float(k_per) <= 1.0:
        raise ValueError("k_per 必须位于 (0, 1]")
    if isinstance(pair_batch_size, bool) or int(pair_batch_size) != pair_batch_size:
        raise TypeError("pair_batch_size 必须是正整数")
    pair_batch_size = int(pair_batch_size)
    if pair_batch_size <= 0:
        raise ValueError("pair_batch_size 必须是正整数")
    device = _as_device(device)

    npatch_total = int(qf_local.shape[2])
    k_selected = max(1, int(math.ceil(npatch_total * float(k_per))))
    all_features = torch.cat([qf_local, gf_local], dim=0)
    all_features = all_features.to(
        device=device, dtype=torch.float32, non_blocking=False
    )
    all_features = all_features.permute(0, 2, 1).contiguous()
    edge_distances = np.empty(left.size, dtype=np.float32)

    _synchronize(device)
    start_time = time.perf_counter()
    cdist_calls = 0
    for pair_start in range(0, left.size, pair_batch_size):
        pair_stop = min(pair_start + pair_batch_size, left.size)
        left_index = torch.from_numpy(
            left[pair_start:pair_stop].astype(np.int64, copy=False)
        ).to(device=device)
        right_index = torch.from_numpy(
            right[pair_start:pair_stop].astype(np.int64, copy=False)
        ).to(device=device)
        left_features = all_features.index_select(0, left_index)
        right_features = all_features.index_select(0, right_index)
        patch_dist = torch.cdist(left_features, right_features, p=2.0)
        cdist_calls += 1
        forward_min = patch_dist.min(dim=-1).values
        backward_min = patch_dist.min(dim=-2).values
        forward_score = _aggregate_patch_minima(forward_min, k_selected)
        backward_score = _aggregate_patch_minima(backward_min, k_selected)
        score = 0.5 * (forward_score + backward_score)
        edge_distances[pair_start:pair_stop] = score.cpu().numpy()
        del (
            left_index,
            right_index,
            left_features,
            right_features,
            patch_dist,
            forward_min,
            backward_min,
            forward_score,
            backward_score,
            score,
        )
        if progress_callback is not None:
            progress_callback("local_pairs", pair_stop, int(left.size))

    _synchronize(device)
    elapsed = time.perf_counter() - start_time
    if not np.all(np.isfinite(edge_distances)):
        raise RuntimeError("稀疏局部距离包含 NaN 或 Inf")
    profile = {
        "local_patch_matching_s": float(elapsed),
        "pair_batch_size": int(pair_batch_size),
        "cdist_calls": int(cdist_calls),
        "npatch_total": int(npatch_total),
        "k_per": float(k_per),
        "k_selected": int(k_selected),
        "matches_per_pair": int(2 * k_selected),
    }
    del all_features
    return edge_distances, profile


def _build_distance_csr(total_num, left, right, edge_distances):
    node_ids = np.arange(total_num, dtype=np.int32)
    source = np.concatenate([left, right, node_ids]).astype(np.int32, copy=False)
    target = np.concatenate([right, left, node_ids]).astype(np.int32, copy=False)
    distance = np.concatenate(
        [edge_distances, edge_distances, np.zeros(total_num, dtype=np.float32)]
    ).astype(np.float32, copy=False)

    order = np.lexsort((target, source))
    source = source[order]
    target = target[order]
    distance = distance[order]
    counts = np.bincount(source, minlength=total_num)
    indptr = np.empty(total_num + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    row_scale = np.zeros(total_num, dtype=np.float32)
    np.maximum.at(row_scale, left, edge_distances)
    np.maximum.at(row_scale, right, edge_distances)
    if np.any(row_scale <= 0.0) or not np.all(np.isfinite(row_scale)):
        raise RuntimeError("局部距离逐行归一化尺度无效")
    normalized = distance / row_scale[source]
    normalized[target == source] = 0.0
    normalized = np.minimum(normalized, np.float32(1.0))
    return indptr, target, normalized.astype(np.float32, copy=False), row_scale


def _initial_sparse_rank(indptr, indices, values, total_num, rank_length):
    initial_rank = np.empty((total_num, rank_length), dtype=np.int32)
    degrees = np.diff(indptr)
    if int(np.min(degrees)) < rank_length:
        raise ValueError(
            "稀疏候选度数不足以执行 k-reciprocal: min_degree={}, required={}".format(
                int(np.min(degrees)), rank_length
            )
        )
    for row in range(total_num):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        row_values = values[start:stop]
        if row_values.size == rank_length:
            selected = np.arange(row_values.size)
        else:
            selected = np.argpartition(row_values, rank_length - 1)[:rank_length]
        selected = selected[np.argsort(row_values[selected])]
        initial_rank[row] = indices[start:stop][selected]
    return initial_rank, degrees


def _lookup_sparse_distances(row, columns, indptr, indices, values):
    columns = np.asarray(columns, dtype=np.int32)
    result = np.full(
        columns.shape, UNKNOWN_NORMALIZED_DISTANCE, dtype=np.float32
    )
    start, stop = int(indptr[row]), int(indptr[row + 1])
    row_indices = indices[start:stop]
    positions = np.searchsorted(row_indices, columns)
    in_range = positions < row_indices.size
    found = np.zeros(columns.shape, dtype=bool)
    if np.any(in_range):
        found[in_range] = row_indices[positions[in_range]] == columns[in_range]
    if np.any(found):
        result[found] = values[start:stop][positions[found]]
    return result


def _build_sparse_v(initial_rank, indptr, indices, values, k1):
    total_num = int(initial_rank.shape[0])
    half_k = int(np.around(k1 / 2.0))
    v_rows, v_columns, v_values = [], [], []
    for row in range(total_num):
        forward = initial_rank[row, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        reciprocal = forward[np.where(backward == row)[0]]
        expansion = reciprocal
        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, : half_k + 1]
            candidate_backward = initial_rank[
                candidate_forward, : half_k + 1
            ]
            candidate_reciprocal = candidate_forward[
                np.where(candidate_backward == candidate)[0]
            ]
            if len(candidate_reciprocal) and len(
                np.intersect1d(candidate_reciprocal, reciprocal)
            ) > (2.0 / 3.0) * len(candidate_reciprocal):
                expansion = np.append(expansion, candidate_reciprocal)
        expansion = np.unique(expansion).astype(np.int32, copy=False)
        distances = _lookup_sparse_distances(
            row, expansion, indptr, indices, values
        )
        weights = np.exp(-distances).astype(np.float32, copy=False)
        weights /= np.sum(weights, dtype=np.float32)
        # The legacy dense V array is float16.  Quantize here, then keep the
        # sparse container float32 because scipy sparse matmul lacks robust
        # float16 support in the project's SciPy version.
        weights = weights.astype(np.float16).astype(np.float32)
        v_rows.append(np.full(expansion.size, row, dtype=np.int32))
        v_columns.append(expansion)
        v_values.append(weights)

    rows = np.concatenate(v_rows)
    columns = np.concatenate(v_columns)
    data = np.concatenate(v_values)
    return scipy_sparse.csr_matrix(
        (data, (rows, columns)), shape=(total_num, total_num), dtype=np.float32
    )


def _sparse_query_expansion(v_matrix, initial_rank, k2):
    if k2 == 1:
        return v_matrix
    total_num = int(initial_rank.shape[0])
    rows = np.repeat(np.arange(total_num, dtype=np.int32), k2)
    columns = initial_rank[:, :k2].reshape(-1)
    data = np.full(rows.size, 1.0 / float(k2), dtype=np.float32)
    averaging = scipy_sparse.csr_matrix(
        (data, (rows, columns)), shape=(total_num, total_num), dtype=np.float32
    )
    expanded = averaging.dot(v_matrix).tocsr()
    expanded.sum_duplicates()
    expanded.data = expanded.data.astype(np.float16).astype(np.float32)
    expanded.eliminate_zeros()
    return expanded


def sparse_local_rerank(
    query_num,
    gallery_num,
    left,
    right,
    edge_distances,
    k1=20,
    k2=6,
    lambda_value=0.3,
    progress_callback=None,
):
    """Run the legacy local k-reciprocal logic on a sparse distance graph."""

    if scipy_sparse is None:
        raise RuntimeError("稀疏 GLSA 需要 scipy: {}".format(_SCIPY_IMPORT_ERROR))
    query_num, gallery_num = int(query_num), int(gallery_num)
    total_num = query_num + gallery_num
    if query_num <= 0 or gallery_num <= 0:
        raise ValueError("query_num 和 gallery_num 必须为正数")
    if k1 <= 0 or k2 <= 0 or k1 >= total_num or k2 > total_num:
        raise ValueError("k1/k2 超出有效范围")
    if not 0.0 <= float(lambda_value) <= 1.0:
        raise ValueError("lambda_value 必须位于 [0, 1]")
    left = np.asarray(left, dtype=np.int32)
    right = np.asarray(right, dtype=np.int32)
    edge_distances = np.asarray(edge_distances, dtype=np.float32)
    if left.shape != right.shape or left.shape != edge_distances.shape:
        raise ValueError("候选边和局部距离数量不一致")
    if edge_distances.size == 0 or not np.all(np.isfinite(edge_distances)):
        raise ValueError("edge_distances 为空或包含 NaN/Inf")

    stage_start = time.perf_counter()
    indptr, indices, normalized, row_scale = _build_distance_csr(
        total_num, left, right, edge_distances
    )
    initial_rank, degrees = _initial_sparse_rank(
        indptr, indices, normalized, total_num, k1 + 1
    )
    v_matrix = _build_sparse_v(
        initial_rank, indptr, indices, normalized, k1
    )
    v_matrix = _sparse_query_expansion(v_matrix, initial_rank, k2)
    v_matrix.sort_indices()
    v_inverse = v_matrix.tocsc()
    v_inverse.sort_indices()

    final_local = np.empty((query_num, gallery_num), dtype=np.float32)
    for query_index in range(query_num):
        temp_min = np.zeros(total_num, dtype=np.float16)
        v_start, v_stop = (
            int(v_matrix.indptr[query_index]),
            int(v_matrix.indptr[query_index + 1]),
        )
        query_columns = v_matrix.indices[v_start:v_stop]
        query_values = v_matrix.data[v_start:v_stop]
        for column, query_value in zip(query_columns, query_values):
            inverse_start = int(v_inverse.indptr[column])
            inverse_stop = int(v_inverse.indptr[column + 1])
            image_indices = v_inverse.indices[inverse_start:inverse_stop]
            image_values = v_inverse.data[inverse_start:inverse_stop]
            contribution = np.minimum(query_value, image_values)
            contribution = contribution.astype(np.float16, copy=False)
            temp_min[image_indices] = temp_min[image_indices] + contribution

        jaccard = 1.0 - temp_min / (2.0 - temp_min)
        original_gallery = np.full(
            gallery_num, UNKNOWN_NORMALIZED_DISTANCE, dtype=np.float32
        )
        row_start, row_stop = (
            int(indptr[query_index]),
            int(indptr[query_index + 1]),
        )
        row_columns = indices[row_start:row_stop]
        row_values = normalized[row_start:row_stop]
        gallery_mask = row_columns >= query_num
        original_gallery[row_columns[gallery_mask] - query_num] = row_values[
            gallery_mask
        ]
        final_local[query_index] = (
            (1.0 - float(lambda_value))
            * jaccard[query_num:].astype(np.float32, copy=False)
            + float(lambda_value) * original_gallery
        )
        if progress_callback is not None:
            progress_callback("local_rerank", query_index + 1, query_num)

    profile = {
        "local_reranking_s": float(time.perf_counter() - stage_start),
        "min_candidate_degree": int(np.min(degrees) - 1),
        "mean_candidate_degree": float(np.mean(degrees - 1)),
        "max_candidate_degree": int(np.max(degrees) - 1),
        "v_nnz": int(v_matrix.nnz),
        "row_scale_min": float(np.min(row_scale)),
        "row_scale_mean": float(np.mean(row_scale)),
        "row_scale_max": float(np.max(row_scale)),
        "unknown_normalized_distance": float(UNKNOWN_NORMALIZED_DISTANCE),
    }
    return final_local, profile


@torch.no_grad()
def glsa(
    probFea,
    galFea,
    qf_local,
    gf_local,
    top_m,
    k1=20,
    k2=6,
    alpha=0.8,
    k_per=0.9,
    pair_batch_size=256,
    calibration_count=16,
    device=None,
    global_context=None,
    return_profile=False,
    progress_callback=None,
):
    """Two-stage GLSA with an all-node sparse Top-M local graph."""

    _validate_global_features(probFea, galFea)
    query_num = int(probFea.shape[0])
    gallery_num = int(galFea.shape[0])
    total_num = query_num + gallery_num
    _validate_local_features(qf_local, gf_local, query_num, gallery_num)
    if isinstance(top_m, bool) or not isinstance(top_m, (int, np.integer)):
        raise TypeError("top_m 必须是整数")
    top_m = int(top_m)
    if top_m < int(k1) or top_m >= total_num:
        raise ValueError("top_m 必须满足 k1 <= top_m < query+gallery")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]")
    if not 0.0 < float(k_per) <= 1.0:
        raise ValueError("k_per 必须位于 (0, 1]")
    device = _as_device(device)

    call_start = time.perf_counter()
    context_reused = global_context is not None
    if global_context is None:
        global_context = prepare_glsa_global_context(
            probFea, galFea, k1=k1, k2=k2
        )
    else:
        _validate_context(global_context, query_num, gallery_num, k1, k2)

    left, right, candidate_profile = build_sparse_candidate_graph(
        global_context.raw_dist,
        top_m=top_m,
        calibration_count=calibration_count,
        device=device,
        progress_callback=progress_callback,
    )
    edge_distances, matching_profile = compute_sparse_edge_local_distances(
        qf_local,
        gf_local,
        left,
        right,
        k_per=k_per,
        pair_batch_size=pair_batch_size,
        device=device,
        progress_callback=progress_callback,
    )
    local_dist, local_profile = sparse_local_rerank(
        query_num,
        gallery_num,
        left,
        right,
        edge_distances,
        k1=int(k1),
        k2=int(k2),
        lambda_value=0.3,
        progress_callback=progress_callback,
    )

    fusion_start = time.perf_counter()
    final_dist = (
        float(alpha) * global_context.reranked_qg
        + (1.0 - float(alpha)) * local_dist
    )
    fusion_s = time.perf_counter() - fusion_start
    if not np.all(np.isfinite(final_dist)):
        raise RuntimeError("稀疏 GLSA 最终距离包含 NaN 或 Inf")

    incremental_s = (
        candidate_profile["candidate_construction_s"]
        + matching_profile["local_patch_matching_s"]
        + local_profile["local_reranking_s"]
        + fusion_s
    )
    global_s = float(global_context.profile["global_postprocess_s"])
    profile = {}
    profile.update(global_context.profile)
    profile.update(candidate_profile)
    profile.update(matching_profile)
    profile.update(local_profile)
    profile.update(
        {
            "top_m": int(top_m),
            "calibration_count": int(calibration_count),
            "alpha": float(alpha),
            "k1": int(k1),
            "k2": int(k2),
            "fusion_s": float(fusion_s),
            "incremental_sparse_s": float(incremental_s),
            "total_postprocess_s": float(global_s + incremental_s),
            "call_wall_s": float(time.perf_counter() - call_start),
            "global_context_reused": bool(context_reused),
        }
    )
    if device.type == "cuda":
        profile["peak_gpu_memory_mb"] = float(
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        )
    else:
        profile["peak_gpu_memory_mb"] = 0.0

    del left, right, edge_distances, local_dist
    if return_profile:
        return final_dist, profile
    return final_dist


__all__ = [
    "SparseGLSAGlobalContext",
    "UNKNOWN_NORMALIZED_DISTANCE",
    "prepare_glsa_global_context",
    "build_sparse_candidate_graph",
    "compute_sparse_edge_local_distances",
    "sparse_local_rerank",
    "glsa",
]
